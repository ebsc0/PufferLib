#include "bandit.h"

#define Env Bandit
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_arms = unpack(kwargs, "num_arms");
    env->reward_std = unpack(kwargs, "reward_std");
    env->drift_std = unpack(kwargs, "drift_std");
    env->history_len = unpack(kwargs, "history_len");
    if (env->history_len <= 0) {
        env->history_len = 1024;
    }
    env->means = calloc(env->num_arms, sizeof(float));
    env->history = calloc(env->history_len, sizeof(float));
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    return 0;
}
