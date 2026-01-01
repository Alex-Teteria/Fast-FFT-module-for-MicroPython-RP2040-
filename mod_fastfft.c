#include "py/runtime.h"
#include "py/objarray.h"

#include "kiss_fft.h"
#include "kiss_fftr.h"

#define FFT_SIZE 512

_Static_assert(sizeof(kiss_fft_scalar) == sizeof(float),
               "kissFFT is not float!");

/* ---------- static buffers ---------- */

static float fft_in[FFT_SIZE];
static kiss_fft_cpx fft_out[FFT_SIZE / 2 + 1];

/* результат (енергія спектра) */
static int32_t spec_buf[FFT_SIZE / 2] __attribute__((aligned(4)));

/* конфігурація kissFFT */
static kiss_fftr_cfg fft_cfg = NULL;
static uint8_t fft_cfg_mem[16384] __attribute__((aligned(8)));

/* ---------- rfft ---------- */

static mp_obj_t fastfft_rfft(mp_obj_t input_obj) {

    /* перевірка типу */
    if (!mp_obj_is_type(input_obj, &mp_type_array)) {
        mp_raise_msg(&mp_type_TypeError,
                     MP_ERROR_TEXT("expected array('h')"));
    }

    mp_obj_array_t *arr = MP_OBJ_TO_PTR(input_obj);

    if (arr->typecode != 'h') {
        mp_raise_msg(&mp_type_TypeError,
                     MP_ERROR_TEXT("expected array('h')"));
    }

    if (arr->len != FFT_SIZE) {
        mp_raise_msg(&mp_type_ValueError,
                     MP_ERROR_TEXT("array length must be 512"));
    }

    int16_t *src = (int16_t *)arr->items;

    /* копіюємо вхід */
    for (int i = 0; i < FFT_SIZE; i++) {
        fft_in[i] = (float)src[i];
    }

    /* ініціалізація FFT (один раз) */
    if (fft_cfg == NULL) {

        size_t mem_needed = 0;
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

    /* FFT */
    kiss_fftr(fft_cfg, fft_in, fft_out);

    /* DC не цікавить */
    spec_buf[0] = 0;

    /* нормалізація: одна операція поза циклом */
    const float norm = 1.0f / (FFT_SIZE * FFT_SIZE);

    for (int k = 1; k < FFT_SIZE / 2; k++) {

        float re = fft_out[k].r;
        float im = fft_out[k].i;

        /* енергія спектра */
        float e = re * re + im * im;

        /* нормалізація */
        e *= norm;

        /* saturation + округлення */
        if (e >= (float)INT32_MAX) {
            spec_buf[k] = INT32_MAX;
        } else {
            spec_buf[k] = (int32_t)(e + 0.5f);
        }
    }

    /* memoryview — БЕЗПЕЧНО */
    return mp_obj_new_memoryview('i', FFT_SIZE / 2, spec_buf);
}

/* ---------- MicroPython glue ---------- */

static MP_DEFINE_CONST_FUN_OBJ_1(
    fastfft_rfft_obj,
    fastfft_rfft
);

static const mp_rom_map_elem_t fastfft_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_fastfft) },
    { MP_ROM_QSTR(MP_QSTR_rfft), MP_ROM_PTR(&fastfft_rfft_obj) },
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
