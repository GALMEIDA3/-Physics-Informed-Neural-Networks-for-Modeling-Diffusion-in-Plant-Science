import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the PINN model
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(20, activation='tanh')  # Reduced number of neurons
        self.dense2 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(1)  # Output layer, representing u(x,t)

    def call(self, x, t):
        inputs = tf.concat([x, t], axis=1)
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

# Loss functions for PDE, boundary, and initial conditions
def loss_pde(model, x, t):
    x = tf.convert_to_tensor(x)
    t = tf.convert_to_tensor(t)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u = model(x, t)
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)
    del tape
    pde_residual = u_t - 0.01 * u_xx  # Example PDE
    return tf.reduce_mean(tf.square(pde_residual))

def loss_boundary(model, x_b, t_b):
    x_b = tf.convert_to_tensor(x_b)
    t_b = tf.convert_to_tensor(t_b)

    u_b = model(x_b, t_b)
    return tf.reduce_mean(tf.square(u_b))

def loss_initial(model, x_i, t_i, u_i):
    x_i = tf.convert_to_tensor(x_i)
    t_i = tf.convert_to_tensor(t_i)
    u_i = tf.convert_to_tensor(u_i)

    u_pred_i = model(x_i, t_i)
    return tf.reduce_mean(tf.square(u_pred_i - u_i))

# Total loss function
def compute_loss(model, x, t, x_boundary, t_boundary, x_initial, u_initial, t_initial):
    pde_loss = loss_pde(model, x, t)
    boundary_loss = loss_boundary(model, x_boundary, t_boundary)
    initial_loss = loss_initial(model, x_initial, t_initial, u_initial)
    total_loss = 10 * pde_loss + boundary_loss + initial_loss  # Adjust weights
    return total_loss

# Optimizer with learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.0)

# Train step with eager execution
def train_step(model, x, t, x_boundary, t_boundary, x_initial, u_initial, t_initial):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, t, x_boundary, t_boundary, x_initial, u_initial, t_initial)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
def train_pinn(epochs=100):
    model = PINN()

    for epoch in range(epochs):
        loss = train_step(model, x_train, t_train, x_boundary, t_boundary, x_initial, u_initial, t_initial)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.numpy()}")

    return model

# Visualization of the solution for fixed time t
def plot_fixed_time(model, t_val):
    x_test = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)
    t_test = np.full_like(x_test, t_val)  # Fixed time value
    u_pred = model(x_test, t_test).numpy()

    plt.plot(x_test, u_pred, label=f"t = {t_val}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(f"Solution u(x,t) at t = {t_val}")
    plt.legend()
    plt.show()

# Animation function to show evolution of u(x, t) over time
def animate_solution(model):
    fig, ax = plt.subplots()
    x_test = np.linspace(0, 1, 100).reshape(-1, 1).astype(np.float32)
    line, = ax.plot(x_test, np.zeros_like(x_test))  # Initial line

    def update(frame):
        t_val = frame / 100  # Time increments from 0 to 1
        t_test = np.full_like(x_test, t_val)
        u_pred = model(x_test, t_test).numpy()
        line.set_ydata(u_pred)
        ax.set_title(f"Solution u(x,t) at t = {t_val:.2f}")
        return line,

    ani = FuncAnimation(fig, update, frames=np.arange(0, 101), interval=100)

    # Save the animation as a video file or GIF
    ani.save("pinn_solution_evolution.gif", writer="imagemagick")

    # Show the animation
    plt.show()

# Example data
x_train = np.random.rand(100, 1).astype(np.float32)
t_train = np.random.rand(100, 1).astype(np.float32)
x_boundary = np.random.rand(20, 1).astype(np.float32)
t_boundary = np.random.rand(20, 1).astype(np.float32)
x_initial = np.random.rand(20, 1).astype(np.float32)
t_initial = np.zeros_like(x_initial)
u_initial = np.sin(np.pi * x_initial).astype(np.float32)

# Train the model
model = train_pinn(epochs=1000)

# Plot for a fixed time t
plot_fixed_time(model, t_val=0.5)

# Create an animation showing the evolution over time
animate_solution(model)
