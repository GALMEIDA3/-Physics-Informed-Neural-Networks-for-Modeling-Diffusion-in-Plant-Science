# Physics-Informed Neural Networks for Modeling Diffusion in Plant Science 

This repository contains a Python implementation of Physics-Informed Neural Networks (PINNs) aimed at modeling diffusion processes relevant to plant science. By integrating neural networks with physical principles, this approach allows for the effective simulation and analysis of diffusion phenomena, which are crucial for understanding various biological processes in plants. 

## Table of Contents 

- [Installation](#installation) 
- [Code Overview](#code-overview) 
- [Model Architecture](#model-architecture) 
- [Loss Functions](#loss-functions) 
- [Training Procedure](#training-procedure) 
- [Visualization](#visualization) 
- [Diffusion PDE and Its Applications in Plant Science](#diffusion-pde-and-its-applications-in-plant-science) 
- [Full Diffusion Equation](#full-diffusion-equation) 
- [Chosen Boundary and Initial Conditions](#chosen-boundary-and-initial-conditions) 
- [Example Data](#example-data) 
- [License](#license) 

## Installation 

To run this code, ensure you have the following dependencies installed: 


pip install numpy tensorflow matplotlib 

## Code Overview
Model Architecture
The model is defined using the PINN class, which inherits from tf.keras.Model. The architecture consists of:

Two hidden layers with 20 neurons each and tanh activation functions.
An output layer that provides the predicted value u(x,t), representing the diffusion process.

class PINN(tf.keras.Model): 
    ...

## Loss Functions

The loss function is composed of three components:

1. PDE Loss: Measures the residual of the diffusion PDE.
2. Boundary Loss: Ensures that boundary conditions are satisfied.
3. Initial Loss: Enforces initial conditions based on the diffusion process.


The total loss is computed as follows:

def compute_loss(...): 
    ... 
## Training Procedure

The training process involves defining an optimizer with a learning rate schedule and a training loop that runs for a specified number of epochs. During each epoch, the gradients are calculated, and the model weights are updated to minimize the total loss.

def train_pinn(epochs=100): 
    ... 
## Visualization

Two visualization functions are included:

1. Plotting for a Fixed Time: Displays the model's prediction of diffusion at a specified time t.

def plot_fixed_time(model, t_val): 
    ... 

2. Animation of Solution Over Time: Creates an animated GIF showing how the diffusion solution evolves as time progresses.

def animate_solution(model): 
    ... 
## Diffusion PDE and Its Applications in Plant Science

The diffusion equation is given by:

$$\frac{\partial u}{\partial x}=D\frac{\partial^2 u}{\partial^2 x}+f(x,t)$$

where:

u(x,t) is the quantity of interest (e.g., concentration), D is the diffusion coefficient, f(x,t) represents an external source term.
In this implementation, we choose f(x,t)=0 to focus on the pure diffusion process without additional external influences.


##  Diffusion Equation
In our code, we implement the diffusion PDE with the external source term f(x,t) set to zero:

$$\frac{\partial u}{\partial x}=D\frac{\partial^2 u}{\partial^2 x}$$
 
This simplification allows us to analyze how diffusion occurs solely due to concentration gradients.

Chosen Boundary and Initial Conditions
We implement the following boundary conditions:

-u(0,t)=0 (Dirichlet boundary condition at x=0),

-u(1,t)=0 (Dirichlet boundary condition at x=1)

For initial conditions, we set:

u(x,0)=g(x)  (the initial concentration profile), which can be chosen based on the specific scenario, such as u(x,0)=sin(πx) for a sinusoidal initial distribution.

Example Data
To train the model, we generate example data as follows:

Random samples for training in the domain [0,1] for both spatial x and time t.

Random samples for boundary conditions and initial conditions based on the defined problem.
This data serves as the input for training the PINN model.

License
This project is licensed under the MIT License - see the LICENSE file for details.















