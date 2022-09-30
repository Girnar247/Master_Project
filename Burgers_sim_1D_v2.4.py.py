from phi.flow import *
from phi.torch.flow import *

##Defining hyperparameters
#math.seed(0)

num_points = 16
max_vel = 50
num_test_steps = 200


BATCH = batch(b=32)
num_train_steps = 3000
num_frames = 11


#math.seed(0)
velocity_ref = CenteredGrid(Noise(BATCH), extrapolation.PERIODIC, x=num_points, bounds=Box(x=(0, max_vel)))*2
diff_ref = math.random_uniform(batch(BATCH), low=0, high=1)
#math.print(velocity_ref)
#math.print(diff)


def burgers_step(v,d, dt=0.1):
    return diffuse.explicit(advect.semi_lagrangian(v, v, dt), d, dt)

velocity_ref_sim = []
velocity_ref_sim_frame = velocity_ref
for frame in range(num_frames):
  velocity_ref_sim.append(velocity_ref_sim_frame)
  velocity_ref_sim_frame = burgers_step(velocity_ref_sim_frame,diff_ref)

reference_sim = math.stack(velocity_ref_sim, batch('frames'))
# vis.plot(reference_sim, animate='frames')
# print(reference_sim.values)
# print(reference_sim.x)
# print(reference_sim.frames)
# print(reference_sim.shape)
# math.print(reference_sim)

#The input of the network would be the fixed velocity sequence (sampled above) and the diffusivity randomly sampled).  The output of the network is supposed to be the physics_loss of all the frames combined for each reference simulation.
net = dense_net(num_points + 1,1,[512,512,512,512,512,512,512], activation='ReLU')
optimizer = adam(net)
parameter_count(net)

def training_loss(X):
  predicted = math.native_call(net,X)
  return math.l2_loss(predicted - physics_loss), X, predicted

#The l2_loss in frame # k of the given simulation in training loop with the reference simulation)
def physics_loss_frame(k):
  return math.l2_loss(reference_sim.frames[k] - velocity_train)

training_loss_list =[]
physics_loss_train_list = []
inputs_train_list = []
predicted_physics_loss_train_list = []
#diffusivities_nn = []
# Here i represents the number of iterations for the neural network during the training process).
for i in range(num_train_steps):
  #velocity_nn_act = []
  physics_loss = 0
  velocity_train = velocity_ref
  diffusivity_nn = math.random_uniform(batch(BATCH), low=0, high=1)
  X_nn = math.stack([*velocity_train.values.x, diffusivity_nn], channel('input_train'))
  inputs_train_list.append(X_nn[num_points])
  #b = math.random_normal(batch(BATCH), channel(vector='rand_diff'))
  for k in range(num_frames): #This represents each frame of the given simulation with a sampled value of diffusivity and given velocity sequence.
    physics_loss = physics_loss + physics_loss_frame(k) #addition of l2_loss for each frame.
    velocity_train = burgers_step(velocity_train, diffusivity_nn) #velocity update scheme after each burgers' step

  loss, _, _ = update_weights(net, optimizer, training_loss, X_nn)
  predicted_physics_loss_train_list.append(math.native_call(net, X_nn))
  physics_loss_train_list.append(physics_loss)
  training_loss_list.append(loss)
  n=num_train_steps/5
  # if i%n==0: print(f"inputs: {X_nn}")
  if i%n==0: print(f"Physics_loss: {physics_loss}")
  if i%n==0: print(f"predicted physics loss: {math.native_call(net,X_nn)}")
  if i%n==0: print(f"Final training_loss: {math.mean(loss)}")


vis.show(CenteredGrid(math.mean(math.stack(training_loss_list, spatial('time')), batch)))




#math.seed(0)
velocity_test = CenteredGrid(Noise(), extrapolation.PERIODIC, x=num_points, bounds=Box(x=(0, max_vel)))*2
diff_test = math.random_uniform(batch(), low=0, high=1)
#diff_test = 0.57
math.print(velocity_test)
math.print(diff_test)


velocity_test_sim = []
velocity_test_sim_frame = velocity_test
for frame in range(num_frames):
  velocity_test_sim.append(velocity_test_sim_frame)
  velocity_test_sim_frame = burgers_step(velocity_test_sim_frame,diff_test)

test_sim = math.stack(velocity_test_sim, batch('frames'))
# vis.plot(test_sim, animate='frames')
# print(test_sim.values)
# print(test_sim.x)
# print(test_sim.frames)
# print(test_sim.shape)
# math.print(test_sim)

#The l2_loss in frame # k of the given simulation in training loop with the reference simulation)
def physics_test_loss_frame(k):
  return math.l2_loss(test_sim.frames[k] - velocity_nn_test)

physics_loss_test_list = []
inputs_test_list = []
predicted_physics_loss_test_list = []
#diffusivities_nn = []
# Here i represents the number of iterations for the neural network during the training process).
for i in range(num_test_steps):
  #velocity_nn_act = []
  physics_test_loss = 0
  #velocity_nn_test =  CenteredGrid(Noise(), extrapolation.PERIODIC, x=6, bounds=Box(x=(0, 20)))*2
  velocity_nn_test = velocity_test
  diffusivity_test = math.random_uniform(batch(), low=0, high=1)
  X_nn_test = math.stack([*velocity_nn_test.values.x, diffusivity_test], channel('input_nn_test'))
  inputs_test_list.append(X_nn_test[num_points])
  for k in range(num_frames): #This represents each frame of the given simulation with a sampled value of diffusivity and given velocity sequence.
    physics_test_loss = physics_test_loss + physics_test_loss_frame(k) #addition of l2_loss for each frame.
    velocity_nn_test = burgers_step(velocity_nn_test,diffusivity_test) #velocity update scheme after each burgers' step
  physics_loss_test_list.append(physics_test_loss)
  predicted_physics_loss_test_list.append(math.native_call(net,X_nn_test))
  #training_losses.append(loss)
  n=num_test_steps/10
  if i%n==0: print(f"inputs_test: {X_nn_test[num_points]}")
  if i%n==0: print(f"Physics_test_loss: {physics_test_loss}")
  if i%n==0: print(f"predicted physics loss: {math.native_call(net,X_nn_test)}")



scatter_test_0 = stack(inputs_test_list + physics_loss_test_list, math.concat_shapes(channel(vector='diffusivity,actual physics loss'), instance(points=num_test_steps)))
vis.show(scatter_test_0)

scatter_test1 = stack(inputs_test_list + predicted_physics_loss_test_list, math.concat_shapes(channel(vector='diffusivity,predicted physics loss'), instance(points=num_test_steps)))
vis.show(scatter_test1)
