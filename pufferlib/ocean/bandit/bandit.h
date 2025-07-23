#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

// Log for aggregation
typedef struct {
    float score; // total reward
    float n;     // step count
} Log;

// Simple multi armed bandit environment
typedef struct {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;

    int k;            // number of arms
    float drift;      // reward drift magnitude
    float* means;     // reward probabilities for each arm

    int tick;
    float total_reward;

    float* history;   // reward total per step for rendering
    int history_len;
} Bandit;

static inline float randf() { return (float)rand() / (float)RAND_MAX; }

void c_reset(Bandit* env) {
    env->tick = 0;
    env->total_reward = 0.0f;
    for (int i=0; i<env->k; i++) {
        env->means[i] = randf();
    }
    if (env->history) {
        memset(env->history, 0, env->history_len*sizeof(float));
    }
    env->observations[0] = 0.0f;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;
    env->log.score = 0.0f;
    env->log.n = 0.0f;
}

void drift_means(Bandit* env) {
    for (int i=0; i<env->k; i++) {
        env->means[i] += env->drift * (2*randf() - 1);
        if (env->means[i] < 0) env->means[i] = 0;
        if (env->means[i] > 1) env->means[i] = 1;
    }
}

void c_step(Bandit* env) {
    env->terminals[0] = 0;
    drift_means(env);
    int a = env->actions[0];
    if (a < 0) a = 0;
    if (a >= env->k) a = env->k - 1;
    float r = randf() < env->means[a] ? 1.0f : 0.0f;
    env->rewards[0] = r;
    env->total_reward += r;
    if (env->tick < env->history_len) {
        env->history[env->tick] = env->total_reward;
    }
    env->tick += 1;
    env->log.score = env->total_reward;
    env->log.n = env->tick;
}

void c_render(Bandit* env) {
    if (!IsWindowReady()) {
        InitWindow(640, 480, "PufferLib Bandit");
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawText(TextFormat("Total Reward: %.0f", env->total_reward), 20, 20, 20, PUFF_WHITE);

    int max_steps = env->history_len;
    float x_scale = 600.0f / max_steps;
    for (int i=1; i<env->tick && i<max_steps; i++) {
        float x1 = 20 + (i-1)*x_scale;
        float x2 = 20 + i*x_scale;
        float y1 = 440 - env->history[i-1]*5;
        float y2 = 440 - env->history[i]*5;
        DrawLine((int)x1, (int)y1, (int)x2, (int)y2, PUFF_RED);
    }
    DrawRectangleLines(20, 240, 600, 200, PUFF_WHITE);
    EndDrawing();
}

void c_close(Bandit* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    free(env->means);
    free(env->history);
}

