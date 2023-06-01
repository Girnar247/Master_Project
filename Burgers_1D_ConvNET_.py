from phi.flow import *
from phi.torch.flow import *
import numpy as np

sim_path = './Burgers_1D_ConvNET_32_test'

num_points = 32
xmin = 0
xmax = 100
num_frames = 7

num_train_steps = 5000
num_total_sims = 8192
num_epochs = 16

diff_min = 0
diff_max = 10

dt = 0.5


def burgers_step(v, d, dt=0.5):
    v = diffuse.explicit(v, d, dt=dt, substeps=3)
    v = advect.semi_lagrangian(v, v, dt=dt)
    return v, d


net = conv_net(2, 1, [128, 128, 128], in_spatial=1 + 1, activation='ReLU')
optimizer = adam(net)
math.print(parameter_count(net))



def training_loss(X):
    predicted = math.native_call(net, X)
    return math.l2_loss(predicted - physics_loss), X, predicted


# for sim_num in range(num_total_sims):
#     math.print(sim_num)
#     sim = Scene.create(sim_path, name='sim')
#     velocity_ref = CenteredGrid(Noise(scale=200, smoothness=0.4), extrapolation.PERIODIC, x=num_points,
#                                 bounds=Box(x=(xmin, xmax))) * 2
#     math.print(velocity_ref)
#     # diff_ref = math.random_uniform(low=0, high=1)
#     # math.print(diff_ref)
#     sim.write(velocity_ref_data=velocity_ref)

sim_path_1 = './Burgers_1D_ConvNET_32_test.pth'
import random

batch_size = 8
BATCH = batch(b=8)
training_loss_list = []

num_indices = list(range(num_total_sims))
math.print(num_indices)

for epoch in range(num_epochs):
    math.print(f"epoch num: {epoch}")
    random.shuffle(num_indices)
    # math.print(num_indices)
    for j in range(0, num_total_sims, batch_size):
        initial_vels = []
        for sim_select in num_indices[j: j + batch_size]:
            # math.print(sim_select)
            sim_read = Scene.at(sim_path,
                                sim_select)  # This part read data from the randomly sampled nth scene from the database each having BATCH number of simulations.
            velocity_ref = sim_read.read('velocity_ref_data')
            # math.print(velocity_ref)
            initial_vels.append(velocity_ref)

        velocity_ref = math.stack(initial_vels, BATCH)
        diff_ref = math.random_uniform(BATCH, low=diff_min, high=diff_max)
        # diff_ref = math.expand(math.random_uniform(BATCH,low=diff_min, high=diff_max), spatial(velocity_ref))
        trj_ref, _ = math.iterate(burgers_step, spatial(time=num_frames - 1), velocity_ref, diff_ref)

        # for steps in range(2):
        # physics_loss = 0
        # math.print(velocity_ref)
        diff_train = math.random_uniform(BATCH, low=diff_min, high=diff_max)
        trj_train, _ = math.iterate(burgers_step, spatial(time=num_frames - 1), velocity_ref, diff_train)
        physics_loss = math.l2_loss(trj_ref - trj_train)
        diffusivity_train = math.expand(diff_train, spatial(trj_ref))
        input_nn = math.stack([trj_ref.values, diffusivity_train], channel(features='state, diffusivity'))

        loss, _, _ = update_weights(net, optimizer, training_loss, input_nn)
        training_loss_list.append(loss)

        if epoch % 2 == 0:
            if j%256==0: print(f"actual: {physics_loss}")
            if j%256==0: print(f"predicted: {math.mean(math.native_call(net, input_nn))}")
            if j%256==0: print(f"training_loss: {loss}")
            save_state(net, sim_path_1)

vis.show(CenteredGrid(math.mean(math.stack(training_loss_list, spatial('steps')), batch)))



save_state(net, sim_path_1)

load_state(net, sim_path_1)

velocity_test = CenteredGrid(Noise(scale=200, smoothness=0.4), extrapolation.PERIODIC, x=num_points,
                             bounds=Box(x=(xmin, xmax))) * 2
math.print(velocity_test)
vis.plot(velocity_test)

diff_ref_test = math.random_uniform( low=diff_min, high=diff_max)
# diff_ref_test = math.expand(math.random_uniform(low=diff_min, high=diff_max), spatial(velocity_test))


trj_ref_test, _ = math.iterate(burgers_step, spatial(time=num_frames - 1), velocity_test, diff_ref_test)
# math.print(velocity_test)
math.print(diff_ref_test)

# vis.plot(trj_ref_test, animate='time')

def surrogate(x):
    diff_x = math.expand(x, spatial(trj_ref_test))
    X_test = math.stack([trj_ref_test.values, diff_x], channel(features='state, diffusivity'))
    predicted_value = math.mean(math.native_call(net, X_test))
    return predicted_value


def gt(x):
    trj_test, _ = math.iterate(burgers_step, spatial(time=num_frames - 1), velocity_test, x)
    return math.l2_loss(trj_ref_test - trj_test)

vis.show(CenteredGrid(lambda diffusivity: math.stack({"surrogate ConvNET": surrogate(diffusivity), "Ground Truth": gt(diffusivity)}, channel('curves')),diffusivity=100, bounds=Box(diffusivity=(diff_min, diff_max))))


def min_gt(x0):
    with math.SolveTape() as tape:
        sol = math.minimize(lambda X: gt(X), Solve('BFGS', 0, 1e-5, x0=x0,suppress=[NotConverged, Diverged]))
        return sol
        #return(plot_fn_act(sol))
        #return sol - x0
        #return tape.solves[0].iterations
        #return tape.solves[0].function_evaluations

def min_surrogate(x0):
    with math.SolveTape() as tape:
        sol = math.minimize(lambda X: surrogate(X), Solve('BFGS', 0, 1e-5, x0=x0,suppress=[NotConverged, Diverged]))
        return sol
        #return(plot_fn_act(sol))
        #return sol - x0
        #return tape.solves[0].iterations
        #return tape.solves[0].function_evaluations

def min_final(x0):
    with math.SolveTape() as tape:
        sol = math.minimize(lambda X: gt(X), Solve('BFGS', 0, 1e-5, x0=min_surrogate(x0),suppress=[NotConverged, Diverged]))
        return sol
        #return(plot_fn_act(sol))
        #return sol - x0
        #return tape.solves[0].iterations
        #return tape.solves[0].function_evaluations



vis.show(CenteredGrid(lambda diffusivity: math.stack({"Ground Truth diffusivity": diff_ref_test,"BFGS Optimization": min_gt(diffusivity),"Final BFGS Optimization": min_final(diffusivity)}, channel('curves')), diffusivity=100, bounds=Box(diffusivity=(diff_min,diff_max))))

math.print(CenteredGrid(lambda diffusivity: min_final(diffusivity), diffusivity=100, bounds=Box(diffusivity=(diff_min,diff_max))))