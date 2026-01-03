#include "py/runtime.h"
#include "py/objarray.h"
#include "py/mperrno.h"

#include "kiss_fft.h"
#include "kiss_fftr.h"

#include <math.h>
#include <stdint.h>

#define FFT_SIZE 512

_Static_assert(sizeof(kiss_fft_scalar) == sizeof(float),
               "kissFFT is not float!");

/* ---------- window support ---------- */
#define PI 3.14159265358979323846f

typedef enum {
    WIN_NONE = 0,
    WIN_HANN,
    WIN_HAMMING,
} win_type_t;

/* Hann and Hamming windows (precomputed on demand) */
static float hann_win[FFT_SIZE];
static float hann_sum_w2 = 0.0f;
static int hann_inited = 0;

static float hamming_win[FFT_SIZE];
static float hamming_sum_w2 = 0.0f;
static int hamming_inited = 0;

static void init_hann_window(void) {
    if (hann_inited) return;
    hann_sum_w2 = 0.0f;
    for (int n = 0; n < FFT_SIZE; n++) {
        float w = 0.5f * (1.0f - cosf(2.0f * PI * n / (FFT_SIZE - 1)));
        hann_win[n] = w;
        hann_sum_w2 += w * w;
    }
    hann_inited = 1;
}

static void init_hamming_window(void) {
    if (hamming_inited) return;
    hamming_sum_w2 = 0.0f;
    for (int n = 0; n < FFT_SIZE; n++) {
        float w = 0.54f - 0.46f * cosf(2.0f * PI * n / (FFT_SIZE - 1));
        hamming_win[n] = w;
        hamming_sum_w2 += w * w;
    }
    hamming_inited = 1;
}

/* ---------- static buffers ---------- */

/* Вхід (float) */
static float fft_in[FFT_SIZE];
/* Вихід RFFT: N/2 + 1 комплексних значень (kiss_fftr використовує цей розмір) */
static kiss_fft_cpx fft_out[FFT_SIZE / 2 + 1];

/* результат (енергія спектра)
   Обчислюємо і повертаємо лише бін[0..N/2-1] але k=0 = 0, Nyquist (N/2) НЕ повертається.
   Розмір = FFT_SIZE/2 (без Nyquist). */
static int32_t spec_buf[FFT_SIZE / 2] __attribute__((aligned(4)));

/* конфігурація kissFFT */
static kiss_fftr_cfg fft_cfg = NULL;
static uint8_t fft_cfg_mem[8192] __attribute__((aligned(8)));

/* ---------- rfft (now accepts optional window param) ---------- */

static mp_obj_t fastfft_rfft(size_t n_args, const mp_obj_t *args) {

    /* args[0] - input buffer
       args[1] - optional window: False (default), True (=> 'hann'), "hann", "hamming" */

    /* Парсимо опційний аргумент */
    win_type_t win_type = WIN_NONE;
    if (n_args == 2) {
        mp_obj_t wobj = args[1];
        /* Правильний порядок перевірок: спочатку рядок, потім явний bool true */
        if (wobj == mp_const_false || wobj == mp_const_none) {
            win_type = WIN_NONE;
        } else if (mp_obj_is_str(wobj)) {
            qstr q = mp_obj_str_get_qstr(wobj);
            if (q == MP_QSTR_hann) {
                win_type = WIN_HANN;
            } else if (q == MP_QSTR_hamming) {
                win_type = WIN_HAMMING;
            } else {
                mp_raise_ValueError(MP_ERROR_TEXT("unsupported window type"));
            }
        } else if (wobj == mp_const_true) {
            /* явне True -> Hann */
            win_type = WIN_HANN;
        } else {
            mp_raise_TypeError(MP_ERROR_TEXT("window must be bool or 'hann'/'hamming'"));
        }
    }

    /* Приймаємо будь-який об'єкт з buffer-protocol довжиною FFT_SIZE * sizeof(int16_t) */
    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(args[0], &bufinfo, MP_BUFFER_READ);

    if (bufinfo.len != FFT_SIZE * (ssize_t)sizeof(int16_t)) {
        mp_raise_msg_varg(&mp_type_ValueError,
                          MP_ERROR_TEXT("input buffer must contain %d int16 elements"),
                          FFT_SIZE);
    }

    /* Трактуємо буфер як int16_t[] */
    int16_t *src = (int16_t *)bufinfo.buf;

    /* копіюємо і конвертуємо в float */
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_in[i] = (float)src[i];
    }

    /* ініціалізація FFT (один раз) -- НЕ потокобезпечно.
       Якщо буде потрібно multi-threaded виклики, додайте mutex або ініціалізуйте при імпорті модуля. */
    if (fft_cfg == NULL) {

        size_t mem_needed = 0;
        /* Запитуємо скільки пам'яті потрібно */
        kiss_fftr_alloc(FFT_SIZE, 0, NULL, &mem_needed);

        if (mem_needed > sizeof(fft_cfg_mem)) {
            mp_raise_msg(&mp_type_RuntimeError,
                         MP_ERROR_TEXT("fft cfg buffer too small"));
        }

        fft_cfg = kiss_fftr_alloc(
            FFT_SIZE,
            0,
            fft_cfg_mem,
            &mem_needed
        );

        if (fft_cfg == NULL) {
            mp_raise_msg(&mp_type_RuntimeError,
                         MP_ERROR_TEXT("kiss_fftr_alloc failed"));
        }
    }

    /* Якщо потрібно - застосовуємо вікно */
    float norm;
    if (win_type == WIN_NONE) {
        norm = 1.0f / (FFT_SIZE * FFT_SIZE);
    } else if (win_type == WIN_HANN) {
        init_hann_window();
        for (int i = 0; i < FFT_SIZE; i++) {
            fft_in[i] *= hann_win[i];
        }
        /* Скоригована нормалізація для збереження енергетичної сумісності */
        norm = 1.0f / (FFT_SIZE * hann_sum_w2);
    } else { /* WIN_HAMMING */
        init_hamming_window();
        for (int i = 0; i < FFT_SIZE; i++) {
            fft_in[i] *= hamming_win[i];
        }
        norm = 1.0f / (FFT_SIZE * hamming_sum_w2);
    }

    /* FFT */
    kiss_fftr(fft_cfg, fft_in, fft_out);

    /* DC не цікавить -> встановлюємо 0 */
    spec_buf[0] = 0;

    /* Заповнюємо лише k = 1 .. N/2 - 1 (Nyquist виключено) */
    for (int k = 1; k < FFT_SIZE / 2; k++) {

        float re = fft_out[k].r;
        float im = fft_out[k].i;

        /* Перевірка компонент перед квадруванням */
        if (!isfinite(re) || !isfinite(im)) {
            spec_buf[k] = 0;
            continue;
        }

        float e = re * re + im * im;

        /* Коротка перевірка після обчислення суми квадратів */
        if (!isfinite(e) || e <= 0.0f) {
            spec_buf[k] = 0;
            continue;
        }

        /* нормалізація */
        e *= norm;

        /* saturation + округлення */
        if (e >= (float)INT32_MAX) {
            spec_buf[k] = INT32_MAX;
        } else {
            spec_buf[k] = (int32_t)(e + 0.5f);
        }
    }

    /* Повертаємо memoryview що посилається на spec_buf.
       Використовуючи існуючу сигнатуру mp_obj_new_memoryview (підтверджено у вашій збірці). */
    return mp_obj_new_memoryview('i', FFT_SIZE / 2, spec_buf);
}

/* ---------- MicroPython glue ---------- */

/* Тепер функція приймає 1..2 аргументи */
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(
    fastfft_rfft_obj,
    1, 2,
    fastfft_rfft
);

static const mp_rom_map_elem_t fastfft_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_fastfft) },
    { MP_ROM_QSTR(MP_QSTR_rfft), MP_ROM_PTR(&fastfft_rfft_obj) },
    /* document supported window names: 'hann', 'hamming' */
};

static MP_DEFINE_CONST_DICT(
    fastfft_module_globals,
    fastfft_module_globals_table
);

const mp_obj_module_t fastfft_user_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *)&fastfft_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_fastfft, fastfft_user_cmodule);
