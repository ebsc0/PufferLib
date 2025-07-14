/* Squared: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

// Standard C library for memory allocation (e.g., malloc, calloc, free) and random numbers (e.g., rand).
#include <stdlib.h>
// Standard C library for memory manipulation functions, especially memset.
#include <string.h>
// A simple and easy-to-use library for creating games and graphical applications. Used here for rendering.
#include "raylib.h"

// --- Action and Observation Constants ---
// Defines integer constants for actions to make the code more readable.
const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

// Defines integer constants for objects on the grid.
const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char TARGET = 2;

// --- Logging Struct ---
// This struct is required by PufferLib's binding system for logging.
// It defines the metrics that will be aggregated from each environment instance
// and sent to logging services like WandB or Neptune.
// Note: The binding system expects these fields to be floats.
typedef struct {
    float perf;           // Recommended: A 0-1 normalized performance metric (e.g., win rate).
    float score;          // Recommended: An unnormalized score metric.
    float episode_return; // Recommended: The sum of all rewards received in an episode.
    float episode_length; // Recommended: The number of steps in an episode.
    // You can add more float fields here for custom logging.
    // The binding file (binding.c) must be updated to handle them.
    float n;              // Required: This must be the last field. It's a counter for averaging logs.
} Log;

// --- Environment State Struct ---
// This struct is required and holds all the state for a single environment instance.
// It is the C equivalent of the Python `PySquared` class.
typedef struct {
    // --- Required PufferLib Fields ---
    Log log;                     // An instance of the Log struct for this environment.
    unsigned char* observations; // Pointer to this env's slice of the observation buffer.
    int* actions;                // Pointer to this env's slice of the action buffer.
    float* rewards;              // Pointer to this env's slice of the reward buffer.
    unsigned char* terminals;    // Pointer to this env's slice of the terminal buffer.
                                 // These pointers are set by PufferLib to point to shared memory.
                                 // The C code reads/writes directly to this memory.

    // --- Custom Environment Fields ---
    // These fields are specific to the Squared environment's logic.
    int size; // The width and height of the grid.
    int tick; // A counter for the number of steps in the current episode.
    int r;    // The agent's current row.
    int c;    // The agent's current column.
} Squared;

// --- Logging Helper Function ---
// A helper function to update the log struct when an episode ends.
void add_log(Squared* env) {
    // Increment performance if the reward is positive (i.e., the agent won).
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    // Add the final reward to the total score.
    env->log.score += env->rewards[0];
    // Add the episode's tick count to the total episode length.
    env->log.episode_length += env->tick;
    // Add the final reward to the total episode return.
    env->log.episode_return += env->rewards[0];
    // Increment the episode counter.
    env->log.n++;
}

// --- Core Environment Functions ---
// These are the C implementations of the environment's logic, equivalent to the
// methods in the Python class. They must be implemented.

// c_reset: Resets the environment to a starting state.
void c_reset(Squared* env) {
    int tiles = env->size * env->size;
    // Use memset for a fast way to set the entire observation grid to EMPTY (0).
    memset(env->observations, 0, tiles * sizeof(unsigned char));

    // Place the agent in the center of the grid.
    env->observations[tiles / 2] = AGENT;
    env->r = env->size / 2;
    env->c = env->size / 2;
    env->tick = 0;

    // Randomly place the target in a location that is not the agent's starting position.
    int target_idx;
    do {
        target_idx = rand() % tiles;
    } while (target_idx == tiles / 2);
    env->observations[target_idx] = TARGET;
}

// c_step: Executes one time step in the environment.
void c_step(Squared* env) {
    env->tick += 1;

    // Read the action for this agent from the action buffer.
    int action = env->actions[0];

    // Reset reward and terminal flags at the start of the step.
    // This is crucial because the buffer memory persists from the previous step.
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    // Erase the agent from its previous position.
    env->observations[env->r * env->size + env->c] = EMPTY;

    // Update the agent's coordinates based on the action.
    if (action == DOWN) {
        env->r += 1;
    } else if (action == RIGHT) {
        env->c += 1;
    } else if (action == UP) {
        env->r -= 1;
    } else if (action == LEFT) {
        env->c -= 1;
    }

    // Check for termination conditions (timeout or out of bounds).
    if (env->tick > 3 * env->size
            || env->r < 0
            || env->c < 0
            || env->r >= env->size
            || env->c >= env->size) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0;
        add_log(env); // Log the episode data.
        c_reset(env); // Reset for the next episode.
        return;       // End the step here.
    }

    // Check if the agent has reached the target.
    int pos = env->r * env->size + env->c;
    if (env->observations[pos] == TARGET) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0;
        add_log(env); // Log the episode data.
        c_reset(env); // Reset for the next episode.
        return;       // End the step here.
    }

    // If the episode is not over, draw the agent in its new position.
    env->observations[pos] = AGENT;
}

// c_render: Renders the environment's current state to a window.
void c_render(Squared* env) {
    // Initialize the window on the first call.
    if (!IsWindowReady()) {
        InitWindow(64 * env->size, 64 * env->size, "PufferLib Squared");
        SetTargetFPS(5); // Slow down rendering to make it watchable.
    }

    // Provide a standard way to exit the render view.
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    // Raylib drawing commands.
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255}); // Dark background

    int px = 64; // Size of each grid cell in pixels.
    for (int i = 0; i < env->size; i++) {
        for (int j = 0; j < env->size; j++) {
            int tex = env->observations[i * env->size + j];
            if (tex == EMPTY) {
                continue;
            }
            // Set color based on the object type.
            Color color = (tex == AGENT) ? (Color){0, 187, 187, 255} : (Color){187, 0, 0, 255};
            DrawRectangle(j * px, i * px, px, px, color);
        }
    }

    EndDrawing();
}

// c_close: Cleans up any resources allocated by the environment.
void c_close(Squared* env) {
    // Close the rendering window if it was opened.
    if (IsWindowReady()) {
        CloseWindow();
    }
    // IMPORTANT: Do not free env->observations, actions, rewards, or terminals here.
    // PufferLib manages that memory.
}