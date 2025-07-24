#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

double randn(double mean, double std) {
    static int has_spare = 0;
    static double spare;
    if (has_spare) {
        has_spare = 0;
        return mean + std * spare;
    }
    has_spare = 1;
    double u, v, s;
    do {
        u = 2.0 * rand() / RAND_MAX - 1.0;
        v = 2.0 * rand() / RAND_MAX - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return mean + std * (u * s);
}

typedef struct {
    float score;
    float n;
} Log;

typedef struct {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int num_arms;
    float reward_std;
    float drift_std;
    float* means;
    float cumulative_reward;
    int step;
    float* history;
    int history_len;
} Bandit;

void c_reset(Bandit* env) {
    for (int i = 0; i < env->num_arms; i++) {
        env->means[i] = randn(0.0, 1.0);
    }
    env->observations[0] = 1.0f;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->cumulative_reward = 0.0f;
    env->step = 0;
    if (env->history_len > 0) {
        memset(env->history, 0, sizeof(float) * env->history_len);
    }
}

void add_log(Bandit* env) {
    env->log.score += env->rewards[0];
    env->log.n += 1.0f;
}

void c_step(Bandit* env) {
    int action = env->actions[0];
    if (action < 0 || action >= env->num_arms) {
        env->rewards[0] = 0.0f;
    } else {
        env->rewards[0] = randn(env->means[action], env->reward_std);
    }
    env->cumulative_reward += env->rewards[0];
    if (env->step < env->history_len) {
        env->history[env->step] = env->cumulative_reward;
    }
    env->step += 1;
    for (int i = 0; i < env->num_arms; i++) {
        env->means[i] += randn(0.0, env->drift_std);
    }
    env->terminals[0] = 0;
    add_log(env);
}

void c_render(Bandit* env) {
    if (!IsWindowReady()) {
        InitWindow(720, 480, "PufferLib Bandit");
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawText(TextFormat("Total Reward: %.2f", env->cumulative_reward), 20, 20, 20, PUFF_WHITE);
    int max_display = env->history_len < env->step ? env->history_len : env->step;
    if (max_display > 1) {
        float sx = 680.0f / (float)(max_display - 1);
        for (int i = 1; i < max_display; i++) {
            float x0 = 20 + sx * (i - 1);
            float y0 = 460 - env->history[i - 1];
            float x1 = 20 + sx * i;
            float y1 = 460 - env->history[i];
            DrawLine((int)x0, (int)y0, (int)x1, (int)y1, PUFF_CYAN);
        }
    }
    EndDrawing();
}

void c_close(Bandit* env) {
    free(env->means);
    free(env->history);
    if (IsWindowReady()) {
        CloseWindow();
    }
}
