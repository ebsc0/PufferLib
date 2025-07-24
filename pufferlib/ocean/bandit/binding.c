#include "bandit.h"

#define Env Bandit
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs){
    env->k = (int)unpack(kwargs, "k");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log){
    assign_to_dict(dict, "score", log->score);
    return 0;
}
