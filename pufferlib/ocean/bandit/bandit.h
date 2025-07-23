#include <stdlib.h>
#include <string.h>
#include "raylib.h"

// Simple non-stationary multi-armed bandit environment

typedef struct {
    float total_reward;
    float n; // required last field
} Log;

typedef struct {
    Log log;                     // required
    float* observations;         // pointer to observation buffer
    int* actions;                // pointer to action buffer
    float* rewards;              // pointer to reward buffer
    unsigned char* terminals;    // pointer to terminal buffer

    int num_arms;                // number of bandit arms
    float* probs;                // reward probabilities per arm
    int step;                    // step counter
} Bandit;

static inline float clamp01(float x) {
    if (x < 0) return 0;
    if (x > 1) return 1;
    return x;
}

void c_reset(Bandit* env) {
    env->step = 0;
    env->log.total_reward = 0;
    env->log.n = 0;
    for (int i = 0; i < env->num_arms; i++) {
        env->probs[i] = (float)rand() / (float)RAND_MAX;
    }
    env->observations[0] = 0; // dummy observation
}

void c_step(Bandit* env) {
    int action = env->actions[0];
    if (action < 0 || action >= env->num_arms) {
        action = 0;
    }
    float r = ((float)rand() / (float)RAND_MAX) < env->probs[action] ? 1.0f : 0.0f;
    env->rewards[0] = r;
    env->terminals[0] = 0;
    env->log.total_reward += r;
    env->log.n += 1;
    env->step += 1;
    env->observations[0] = 0; // constant observation

    for (int i = 0; i < env->num_arms; i++) {
        float noise = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.02f;
        env->probs[i] = clamp01(env->probs[i] + noise);
    }
}

void c_render(Bandit* env) {
    if (!IsWindowReady()) {
        InitWindow(640, 480, "PufferLib Bandit");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("Step: %d", env->step), 20, 20, 20, (Color){241,241,241,255});
    DrawText(TextFormat("Total Reward: %.1f", env->log.total_reward), 20, 50, 20, (Color){241,241,241,255});
    EndDrawing();
}

void c_close(Bandit* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    free(env->probs);
}
