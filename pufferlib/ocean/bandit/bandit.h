#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

typedef struct {
    float score;
    float n; // Required last field
} Log;

typedef struct {
    Log log;
    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int num_arms;
    int horizon;
    float* means;
    float* history;
    int step;
    float episode_return;
} Bandit;

void c_reset(Bandit* env) {
    env->step = 0;
    env->episode_return = 0.0f;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;
    for(int i = 0; i < env->num_arms; i++) {
        env->means[i] = 0.0f;
    }
}

void c_step(Bandit* env) {
    int action = env->actions[0];
    if(action < 0) action = 0;
    if(action >= env->num_arms) action = env->num_arms - 1;

    float noise = ((float)rand()/(float)RAND_MAX - 0.5f);
    float reward = env->means[action] + noise;
    env->rewards[0] = reward;
    env->episode_return += reward;
    if(env->step < env->horizon)
        env->history[env->step] = env->episode_return;
    env->step += 1;

    for(int i=0;i<env->num_arms;i++) {
        env->means[i] += ((float)rand()/(float)RAND_MAX - 0.5f) * 0.01f;
    }

    env->terminals[0] = 0;
    if(env->step >= env->horizon) {
        env->terminals[0] = 1;
        env->log.score += env->episode_return / env->horizon;
        env->log.n += 1;
        c_reset(env);
    }

    env->observations[0] = 0.0f;
}

void c_render(Bandit* env) {
    if(!IsWindowReady()) {
        InitWindow(720, 480, "PufferLib Bandit");
        SetTargetFPS(30);
    }

    if(IsKeyDown(KEY_ESCAPE))
        exit(0);

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawText(TextFormat("Total Reward: %.2f", env->episode_return), 10, 10, 20, PUFF_WHITE);
    for(int i=1;i<env->step && i<env->horizon;i++) {
        DrawLine(50+i-1, 440 - (int)(env->history[i-1]*10), 50+i, 440 - (int)(env->history[i]*10), PUFF_CYAN);
    }
    EndDrawing();
}

void c_close(Bandit* env) {
    if(IsWindowReady())
        CloseWindow();
    free(env->means);
    free(env->history);
}
