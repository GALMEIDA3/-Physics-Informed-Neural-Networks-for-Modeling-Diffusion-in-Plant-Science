# Physics-Informed Neural Networks for Modeling Diffusion in Plant Science 

This repository contains a Python implementation of Physics-Informed Neural Networks (PINNs) aimed at modeling diffusion processes relevant to plant science. By integrating neural networks with physical principles, this approach allows for the effective simulation and analysis of diffusion phenomena, which are crucial for understanding various biological processes in plants. 

## Table of Contents 

- [Installation](#installation) 
- [Usage](#usage) 
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

```bash 
pip install numpy tensorflow matplotlib 
Usage
Clone the repository:

bash
Copy code
git clone <repository-url> 
cd <repository-directory> 
Run the script:

bash
Copy code
python <script-name>.py 
The script will train the PINN model and produce plots of the diffusion solution at a fixed time, along with an animation showing the evolution of the solution over time.

Code Overview
Model Architecture
The model is defined using the PINN class, which inherits from tf.keras.Model. The architecture consists of:

∂u/∂t = D ∂²u/∂x² + f(x,t) 


Two hidden layers with 20 neurons each and tanh activation functions.
An output layer that provides the predicted value 
𝑢
(
𝑥
,
𝑡
)
u(x,t), representing the diffusion process.
python
Copy code
class PINN(tf.keras.Model): 
    ... 
Loss Functions
The loss function is composed of three components:

PDE Loss: Measures the residual of the diffusion PDE.
Boundary Loss: Ensures that boundary conditions are satisfied.
Initial Loss: Enforces initial conditions based on the diffusion process.
The total loss is computed as follows:

python
Copy code
def compute_loss(...): 
    ... 
Training Procedure
The training process involves defining an optimizer with a learning rate schedule and a training loop that runs for a specified number of epochs. During each epoch, the gradients are calculated, and the model weights are updated to minimize the total loss.

python
Copy code
def train_pinn(epochs=100): 
    ... 
Visualization
Two visualization functions are included:

Plotting for a Fixed Time: Displays the model's prediction of diffusion at a specified time 
𝑡
t.
python
Copy code
def plot_fixed_time(model, t_val): 
    ... 
Animation of Solution Over Time: Creates an animated GIF showing how the diffusion solution evolves as time progresses.
python
Copy code
def animate_solution(model): 
    ... 
Diffusion PDE and Its Applications in Plant Science
The diffusion partial differential equation (PDE) describes how substances such as nutrients, water, and gases spread out over time within a medium. The most common form of the diffusion equation is given by:

scss
Copy code
∂u/∂t = D ∂²u/∂x² + f(x,t) 
where 
𝑢
(
𝑥
,
𝑡
)
u(x,t) is the quantity of interest (e.g., concentration), 
𝐷
D is the diffusion coefficient, and 
𝑓
(
𝑥
,
𝑡
)
f(x,t) represents an external source term. In this implementation, we choose 
𝑓
(
𝑥
,
𝑡
)
=
0
f(x,t)=0 to focus on the pure diffusion process without additional external influences.

Full Diffusion Equation
In our code, we implement the diffusion PDE with the external source term 
𝑓
(
𝑥
,
𝑡
)
f(x,t) set to zero:

mathematica
Copy code
∂u/∂t = D ∂²u/∂x² 
This simplification allows us to analyze how diffusion occurs solely due to concentration gradients.

Chosen Boundary and Initial Conditions
We implement the following boundary conditions:

𝑢
(
0
,
𝑡
)
=
𝑢
0
u(0,t)=u 
0
​
  (Dirichlet boundary condition at 
𝑥
=
0
x=0)
𝑢
(
1
,
𝑡
)
=
𝑢
𝐿
u(1,t)=u 
L
​
  (Dirichlet boundary condition at 
𝑥
=
1
x=1)
For initial conditions, we set:

𝑢
(
𝑥
,
0
)
=
𝑢
𝑖
(
𝑥
)
u(x,0)=u 
i
​
 (x) (the initial concentration profile), which can be chosen based on the specific scenario, such as 
𝑢
𝑖
(
𝑥
)
=
sin
⁡
(
𝜋
𝑥
)
u 
i
​
 (x)=sin(πx) for a sinusoidal initial distribution.
Example Data
To train the model, we generate example data as follows:

Random samples for training in the domain 
[
0
,
1
]
[0,1] for both spatial 
𝑥
x and time 
𝑡
t.
Random samples for boundary conditions and initial conditions based on the defined problem.
This data serves as the input for training the PINN model.

License
This project is licensed under the MIT License - see the LICENSE file for details.
