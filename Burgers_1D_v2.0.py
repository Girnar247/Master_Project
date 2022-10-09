from phi.flow import *
from phi.torch.flow import *

BATCH = batch(sim=32)
num_points = 8
num_frames = 11
max_vel = 30
num_train_steps = 35000
num_test_steps = 200

# math.seed(0)
velocity_ref = CenteredGrid(Noise(BATCH), extrapolation.PERIODIC, x=num_points, bounds=Box(x=(0, max_vel))) * 5
diffusivity_ref = math.random_uniform(batch(BATCH), low=0.5, high=0.6)
#math.print(velocity_ref)
#math.print(diffusivity_ref)


def burgers_step(v, d, dt=0.1):
    return diffuse.explicit(advect.semi_lagrangian(v, v, dt), d, dt)


velocity_ref_frames = []
# velocity_ref_sim = velocity_ref
# diffusivity = diffusivity_ref
velocity_ref_sim = velocity_ref
for frame in range(num_frames):
    # math.print(velocity_ref_sim)
    # math.print(diffusivity_ref)
    velocity_ref_frames.append(velocity_ref_sim.values)
    velocity_ref_sim = burgers_step(velocity_ref_sim, diffusivity_ref)
    # velocity_ref_frames

ref_sim = math.stack(velocity_ref_frames, channel('frames'))
#math.print(ref_sim)
# ref_sim_ = math.stack(velocity_ref_frames, channel('inputs'))

# math.print(diffusivity_ref)
# print(ref_sim.values)
# print(ref_sim.x)
# print(ref_sim.frames)
# print(ref_sim.shape)
# math.print(ref_sim)

# math.print(ref_sim.frames[0])  # This represents the starting values of the sampled velocity)
# math.print(ref_sim.frames[num_frames - 1])  # This represents the last frame of the reference simulation)

# The input of the network would be the fixed velocity sequence (sampled above) and the diffusivity randomly sampled).  The output of the network is supposed to be the physics_loss of all the frames combined for each reference simulation.
hidden_size = num_points*num_frames + 1
net = dense_net(hidden_size, 1, [64,128,512,256], activation='ReLU')
optimizer = adam(net)
math.print(parameter_count(net))


def training_loss(X):
    predicted = math.native_call(net, X)
    return math.l2_loss(predicted - physics_loss), X, predicted


# The l2_loss in frame # k of the given simulation in training loop with the reference simulation)
def physics_loss_frame(k):
    return math.l2_loss(ref_sim.frames[k] - velocity_train)


# math.print(velocity_ref)

inputs_train_list = []
inputs_diffusivity_train_list = []
physics_loss_train_list = []
predicted_physics_loss_train_list = []
training_loss_list = []
training_performance = []
# Here i represents the number of iterations for the neural network during the training process).
for i in range(num_train_steps):
    physics_loss = 0
    velocity_train = velocity_ref
    diffusivity_train = math.random_uniform(batch(BATCH), low=0, high=1)
    # print(velocity_ref.values.x, diffusivity_train)
    # X_train = math.stack([*velocity_ref.values.x, diffusivity_train], channel('input_train')) ## Rethink about this
    X_train = velocity_ref
    # X_train = math.stack([*sum(r.x for r in ref_sim), ()), diffusivity_train], channel('input_train'))
    # b = math.random_normal(batch(BATCH), channel(vector='rand_diff')) #Random added noise, if required

    for k in range(num_frames - 1):  # This represents each frame of the given simulation with a sampled value of diffusivity and given velocity sequence.
        physics_loss = physics_loss + physics_loss_frame(k)  # addition of l2_loss for each frame.
        if k < num_frames -1:
            velocity_train = burgers_step(velocity_train,diffusivity_train)  # velocity update scheme after each burgers' step
            # print(velocity_train.values.x, X_train)
            X_train = math.concat([velocity_train, X_train], channel('x'))
            # print(X_train)

    X_train = math.stack([*X_train.values.x, diffusivity_train], channel('input_train'))

    inputs_train_list.append(X_train[num_points])
    inputs_diffusivity_train_list.append(diffusivity_train)

    loss, _, _ = update_weights(net, optimizer, training_loss, X_train)
    physics_loss_train_list.append(physics_loss)
    predicted_physics_loss_train_list.append(math.native_call(net, X_train))
    training_performance.append(physics_loss - math.native_call(net, X_train))
    training_loss_list.append(loss)

    n = num_train_steps / 5
    if i % n == 0: print(f"inputs: {X_train}")
    # if i%n==0: print(f"input_diffusivity: {diffusivity_train}")
    if i % n == 0: print(f"Physics_loss: {physics_loss}")
    if i % n == 0: print(f"predicted physics loss: {math.native_call(net, X_train)}")
    if i % n == 0: print(f"training_loss: {loss}")

vis.show(CenteredGrid(math.mean(math.stack(training_loss_list, spatial('time')), batch)))

# math.seed(0)
velocity_test = CenteredGrid(Noise(), extrapolation.PERIODIC, x=num_points, bounds=Box(x=(0, max_vel)))*5
diff_test = math.random_uniform(batch(), low=0.5, high=0.6)
math.print(diff_test)
math.print(velocity_test)

test_sim_frames = []
velocity_test_sim_frame = velocity_test
for frame in range(num_frames):
    #math.print(velocity_test_sim_frame)
    test_sim_frames.append(velocity_test_sim_frame)
    velocity_test_sim_frame = burgers_step(velocity_test_sim_frame, diff_test)

test_sim = math.stack(test_sim_frames, batch('frames'))

# print(test_sim.values)
# print(test_sim.x)
# print(test_sim.frames)
# print(test_sim.shape)

# math.print(test_sim_frames[num_frames - 1])

# vis.plot(test_sim, animate='frames')


def physics_loss_test_frame(k):
    return math.l2_loss(test_sim.frames[k] - velocity_nn_test)


inputs_test_list = []
inputs_diffusivity_test_list = []
physics_loss_test_list = []
predicted_physics_loss_test_list = []
training_loss_list = []
training_performance = []
# Here i represents the number of iterations for the neural network during the training process).
for i in range(num_test_steps):
    physics_test_loss = 0
    velocity_nn_test = velocity_test
    diffusivity_test = math.random_uniform(batch(), low=0, high=1)
    # print(velocity_ref.values.x, diffusivity_train)
    # X_train = math.stack([*velocity_ref.values.x, diffusivity_train], channel('input_train')) ## Rethink about this
    X_test = velocity_nn_test
    # X_train = math.stack([*sum(r.x for r in ref_sim), ()), diffusivity_train], channel('input_train'))
    # b = math.random_normal(batch(BATCH), channel(vector='rand_diff')) #Random added noise, if required

    for k in range(
            num_frames - 1):  # This represents each frame of the given simulation with a sampled value of diffusivity and given velocity sequence.
        physics_test_loss = physics_test_loss + physics_loss_test_frame(k)  # addition of l2_loss for each frame.
        if k < num_frames -1:
            velocity_nn_test = burgers_step(velocity_nn_test,diffusivity_test)  # velocity update scheme after each burgers' step
            # print(velocity_nn_test.values.x, X_test)
            X_test = math.concat([velocity_nn_test, X_test], channel('x'))
            # print(X_test)

    X_test = math.stack([*X_test.values.x, diffusivity_test], channel('input_test'))

    inputs_test_list.append(X_test)
    inputs_diffusivity_test_list.append(diffusivity_test)

    physics_loss_test_list.append(physics_test_loss)
    predicted_physics_loss_test_list.append(math.native_call(net, X_test))
    # training_performance.append(physics_loss - math.native_call(net, X_train))
    # training_loss_list.append(loss)

    n = num_test_steps / 5
    if i % n == 0: print(f"inputs: {X_test}")
    # if i%n==0: print(f"input_diffusivity: {X_test[45]}")
    if i % n == 0: print(f"Physics_loss: {physics_test_loss}")
    if i % n == 0: print(f"predicted physics loss: {math.native_call(net, X_test)}")
    # if i%n==0: print(f"training_loss: {loss}")


scatter_test = stack(inputs_diffusivity_test_list + predicted_physics_loss_test_list,
                     math.concat_shapes(channel(vector='diffusivity,predicted physics Loss'),
                                        instance(points=num_test_steps)))
vis.show(scatter_test)

scatter_actual = stack(inputs_diffusivity_test_list + physics_loss_test_list,
                       math.concat_shapes(channel(vector='diffusivity, physics Loss'), instance(points=num_test_steps)))
vis.show(scatter_actual)

