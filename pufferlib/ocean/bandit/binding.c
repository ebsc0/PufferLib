#include "bandit.h"

#define Env Bandit
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->k = (int)unpack(kwargs, "k");
    env->drift = unpack(kwargs, "drift");
    env->history_len = (int)unpack(kwargs, "history_size");
    env->means = (float*)calloc(env->k, sizeof(float));
    env->history = (float*)calloc(env->history_len, sizeof(float));
    if (!env->means || !env->history) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate bandit arrays");
        return -1;
    }
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
