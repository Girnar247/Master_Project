# from phi.tf.flow import *
from phi.torch.flow import *
# from phi.jax.stax.flow import *
from phi._troubleshoot import plot_solves

#parameters
x0 = vec(x=0.4, y=0.6)
# speed = 1
alpha = 0
num_key_states = 15
num_trj_frames = 60
layers = 4 #billiards_layers
num_input_dense = 2*(num_key_states + 1)*((1 + 2 + 3 + 4) + 1)
fixed_goal = vec(x=2.4, y=0.5)
parm_1_min = 0.5
parm_1_max =  2
# parm_2_min = 0.8
# parm_2_max =  2
noise_scale_1= 0.18
noise_scale_2= 0.28
skew_scale_high= 3
skew_scale_low= 0.1
num_train_steps = 10


def billiards_triangle(billiard_layers=layers, radius=.03):
    coords = []
    for i in range(billiard_layers):
        for j in range(i + 1):
            coords.append(vec(x=i * 2.4 * radius + 0.6, y=j * 2.4 * radius + 0.6 - i * radius * 0.7))
    return Sphere(stack(coords, instance('balls')), radius=radius)

show(billiards_triangle())
# math.print(billiards_triangle())

def physics_step(v: PointCloud, time: float or Tensor, elasticity=.8, friction=0.5):
  other_points = rename_dims(v.points, 'balls', 'others') #Another dimension 'others' to compute pairwise quantities.
  rel_v = v.values - rename_dims(v.values, 'balls', 'others') #Pairwise relative velocities of balls
  t_to_closest = (other_points - v.points).vector * rel_v.vector / math.vec_squared(rel_v)  #assuming linear velocity
  closest = v.points + t_to_closest * rel_v #Are these coordinates to closest approach from ball i to ball j. If collision occurs then it is the coordinates of collision? WHy have we added v.points?
  pass_by_distance_squared = math.vec_squared(closest - other_points)  # will impact if < 2 R, otherwise neg_offset will be NaN
  radius_sum = v.elements.radius + rename_dims(v.elements.radius, 'balls', 'others') #This will always be 0.12?
  impact_offset = math.sqrt(radius_sum ** 2 - pass_by_distance_squared)  #positive, distance by how much the impact happens before the closest point on the line
  impact_time_no_friction = t_to_closest - impact_offset / math.vec_length(rel_v, eps=1e-5)  #assuming linear velocity
  impact_time = - math.log(1 - friction * impact_time_no_friction) / friction
  impact_time = math.where(impact_time < 1e-3, NAN, impact_time)
  first_impact_time = math.finite_min(impact_time, default=INF)
  friction_factor = math.exp(- first_impact_time * friction)
  has_impact = impact_time <= first_impact_time + 1e-3  #Handle simultaneous collisions in one go
  impact_relative_position = other_points - (v.points + impact_time_no_friction * rel_v)
  rel_v_at_impact = rel_v * friction_factor
  impulse = -(1 + elasticity) * .5 * (rel_v_at_impact.vector * impact_relative_position.vector) * impact_relative_position / math.vec_squared(impact_relative_position)
  travel_distance = v.values / friction * (1 - friction_factor)
  v = v.with_elements(v.elements.shifted(travel_distance))  #Update position
  v *= friction_factor  #Deceleration due to friction
  v += math.finite_sum(math.where(has_impact, impulse, 0), 'others')  #transfer momentum in case collision occurs
  return v, time + first_impact_time #physics_step returns v after each iteration.


def sample_linear_trajectory(states: PointCloud, times: Tensor, time_dim: math.Shape, velocity_threshold=.1, friction=.5):
  max_velocity = math.max(math.vec_length(states.keys[0].values))
  max_time = math.log(max_velocity / velocity_threshold) / friction
  indices = math.range(spatial('keys'), states.keys.size)
  lin_t = math.linspace(0, max_time, time_dim)
  key_i = math.max(math.where(times <= lin_t, indices, -1), 'keys')
  prev_vel = states.values.keys[key_i]
  prev_pos = states.points.keys[key_i]
  prev_time = times.keys[key_i]
  dt = lin_t - prev_time
  friction_factor = math.exp(- dt * friction)
  travel_distance = prev_vel / friction * (1 - friction_factor)
  new_pos = prev_pos + travel_distance
  new_velocities = prev_vel * friction_factor
  return PointCloud(Sphere(new_pos, radius=states.elements.radius[{'keys': 0}]), new_velocities), lin_t


net = conv_net(3, 1, [64,128,64], activation='ReLU', in_spatial=2)
optimizer = adam(net)
math.print(parameter_count(net))
net2 = conv_net(3, 1, [64,128,64], activation='ReLU', in_spatial=2)
optimizer2 = adam(net2)
math.print(parameter_count(net2))
net3 = conv_net(3, 1, [64,128,64], activation='ReLU', in_spatial=2)
optimizer3 = adam(net3)
# net4 = conv_net(3, 1, [64,128,64], activation='ReLU', in_spatial=2)
# optimizer4 = adam(net4)
# net5 = conv_net(3, 1, [64,128,64], activation='ReLU', in_spatial=2)
# optimizer5 = adam(net5)


def training_loss(X):
    predicted = math.native_call(net, X)
    return math.l2_loss(predicted - physics_loss), X, predicted
def training_loss_noise_1(X):
    predicted = math.native_call(net2, X)
    return math.l2_loss(predicted - physics_loss), X, predicted
def training_loss_noise_1_rf(X):
    predicted = math.native_call(net3, X)
    loss_scale = math.where(predicted > physics_loss, skew_scale_high, skew_scale_low)
    return math.l2_loss(predicted - physics_loss)*loss_scale, X, predicted
# def training_loss_noise_2(X):
#     predicted = math.native_call(net4, X)
#     return math.l2_loss(predicted - physics_loss), X, predicted
# def training_loss_noise_2_rf(X):
#     predicted = math.native_call(net5, X)
#     loss_scale = math.where(predicted > physics_loss, skew_scale_high, skew_scale_low)
#     return math.l2_loss(predicted - physics_loss)*loss_scale, X, predicted


#Training Loop


# training loop

training_loss_list = []

for steps in range(num_train_steps):
    x_init = x0
    balls = billiards_triangle()
    cue_ball = Sphere(tensor([x_init], instance('balls'), channel(vector='x,y')), radius=.03)
    all_balls = math.concat([cue_ball, balls], instance('balls'))
    # balls_v = PointCloud(balls, tensor([(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)], shape(balls)))
    balls_v = PointCloud(balls, tensor([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], shape(balls)))
    # v0_speed = math.random_uniform(low=0.4, high=1)
    #v0_speed = speed
    #v0_speed_train = speed
    v0_speed = math.random_uniform(batch(b=8), low=parm_1_min, high=parm_1_max)
    v0_speed_train = math.random_uniform(batch(b=8), low=parm_1_min, high=parm_1_max)
    #alpha = math.random_uniform(batch(b=8), low=parm_1_min, high=parm_1_max)
    alpha_train = alpha
    v0 = vec(x=v0_speed * math.cos(alpha), y=v0_speed * math.sin(alpha))
    v0_train = vec(x=v0_speed_train * math.cos(alpha_train), y=v0_speed_train * math.sin(alpha_train))
    noise = math.random_normal(batch(b=8))
    # print(f"alpha: {alpha}")
    # math.print(alpha)
    # print(f"v0: {v0}")
    # math.print(v0)
    cue_ball_v = PointCloud(cue_ball, tensor([v0], shape(cue_ball)))
    cue_ball_v_train = PointCloud(cue_ball, tensor([v0_train], shape(cue_ball)))
    # print(f"cue_ball_v: {cue_ball_v}")
    # math.print(cue_ball_v)
    all_balls_v = math.concat([cue_ball_v, balls_v], instance('balls'))
    all_balls_v_train = math.concat([cue_ball_v_train, balls_v], instance('balls'))

    # print(f"all_balls_v: {all_balls_v}")
    # math.print(all_balls_v)
    key_states, key_times = iterate(physics_step, spatial(keys=num_key_states), all_balls_v, 0)
    key_states_train, key_times_train = iterate(physics_step, spatial(keys=num_key_states), all_balls_v_train, 0)
    # print(f"key_states: {key_states}")
    # math.print(key_states)
    # print(f"key_times: {key_times}")
    # math.print(key_times)
    trj_ref, _ = sample_linear_trajectory(key_states, key_times, spatial(t=num_trj_frames))
    trj_train, _ = sample_linear_trajectory(key_states_train, key_times_train, spatial(t=num_trj_frames))  # Batch dimension aate hi is line me problem hai
    # math.print(trj.balls[-1].t[-1].balls[-1])
    # trj_train = physics_loss(x0, v0, goal)[1]
    # # math.print(trj_train.t[::6].values.shape)
    # math.print(physics_loss(x0, v0, goal)[0])
    physics_loss = math.l2_loss(trj_train.elements.center - trj_ref.elements.center)
    # math.print(physics_loss)
    noise = math.random_normal(batch(b=8))
    input_nn = math.rename_dims(trj_ref.t[::6].elements.center, instance('balls'), spatial('balls'))
    #input_nn_v = math.rename_dims(trj_ref.values, instance('balls'), spatial('balls'))
    input_speed = math.expand(v0_speed_train, shape(input_nn.vector[0]))
    # math.print(input_speed)
    #input_alpha = math.expand(alpha_train, shape(input_nn.vector[0]))
    input_nn_conv = math.stack([input_nn.vector[0], input_nn.vector[1], input_speed], channel(features='input_pos_x, input_pos_y,input_speed'))
    # math.print(input_nn_conv)
    input_nn_conv_noise_1 = math.stack([input_nn.vector[0], input_nn.vector[1], input_speed + noise_scale_1 * noise], channel(features='input_pos_x, input_pos_y, input_speed'))
    #input_nn_conv_noise_2 = math.stack([input_nn.vector[0], input_nn.vector[1],  input_speed+ noise_scale_2 * noise], channel(features='input_pos_x, input_pos_y,input_speed'))

    loss, _, _ = update_weights(net, optimizer, training_loss, input_nn_conv)
    loss2, _, _ = update_weights(net2, optimizer2, training_loss_noise_1, input_nn_conv_noise_1)
    loss3, _, _ = update_weights(net3, optimizer3, training_loss_noise_1_rf, input_nn_conv_noise_1)
    #oss4, _, _ = update_weights(net4, optimizer4, training_loss_noise_2, input_nn_conv_noise_2)
   #loss5, _, _ = update_weights(net5, optimizer5, training_loss_noise_2_rf, input_nn_conv_noise_2)

    training_loss_list.append(loss)

    k = num_train_steps / 10
    if steps % k == 0:
        print(f"Physics_loss: {physics_loss}")
        print(f"Predicted_NN: {math.mean(math.native_call(net, input_nn_conv)), math.mean(math.native_call(net2, input_nn_conv_noise_1)), math.mean(math.native_call(net3, input_nn_conv_noise_1))}")
        print(f"Training_loss_NN: {loss, loss2, loss3}")
        save_state(net, './billiards_1D_ConvNet_3.1_23032023_01.pth')
        save_state(net2, './billiards_1D_ConvNet_3.1_23032023_noise_1_01.pth')
        save_state(net3, './billiards_1D_ConvNet_3.1_23032023_noise_1_rf_01.pth')
        #save_state(net4, './billiards_1D_ConvNet_3.1_23032023_noise_2_01.pth')
        #save_state(net5, './billiards_ConvNet_3.1_23032023_noise_2_rf_01.pth')

save_state(net, './billiards_1D_ConvNet_3.1_23032023_01.pth')
save_state(net2, './billiards_1D_ConvNet_3.1_23032023_noise_1_01.pth')
save_state(net3, './billiards_1D_ConvNet_3.1_23032023_noise_1_rf_01.pth')
#save_state(net4, './billiards_1D_ConvNet_3.1_23032023_noise_2_01.pth')
#save_state(net5, './billiards_ConvNet_3.1_23032023_noise_2_rf_01.pth')

vis.show(CenteredGrid(math.mean(math.stack(training_loss_list, spatial('steps')), batch)))



alpha_test = alpha
speed_test = math.random_uniform(batch(low=parm_1_min, high=parm_1_max))
math.print(speed_test)
v0_test = vec(x=speed_test * math.cos(alpha_test), y=speed_test * math.sin(alpha_test))
cue_ball_v_test = PointCloud(cue_ball, tensor([(v0_test)], shape(cue_ball)))
all_balls_v_test = math.concat([cue_ball_v_test, balls_v], instance('balls'))
key_states_test, key_times_test = iterate(physics_step, spatial(keys=num_key_states), all_balls_v_test, 0)
trj_test, _ = sample_linear_trajectory(key_states_test, key_times_test, spatial(t=num_trj_frames))





#
# def landscape(v0_speed):
#   balls = billiards_triangle()
#   x0_test = x0
#   #goal = fixed_goal
#   cue_ball = Sphere(tensor([x0_test], instance('balls'), channel(vector='x,y')), radius=.03)
#   all_balls = math.concat([cue_ball, balls], instance('balls'))
#   #balls_v = PointCloud(balls, tensor([(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)], shape(balls)))
#   balls_v = PointCloud(balls, tensor([(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)], shape(balls)))
#   #v0_speed = speed
#   # math.print(alpha_test)
#   # math.print(v0_speed)
#   alpha_test = alpha
#   v0 = vec(x=v0_speed * math.cos(alpha), y=v0_speed * math.sin(alpha))
#   cue_ball_v = PointCloud(cue_ball, tensor([(v0)], shape(cue_ball)))
#   all_balls_v = math.concat([cue_ball_v, balls_v], instance('balls'))
#   key_states, key_times = iterate(physics_step, spatial(keys=num_key_states), all_balls_v, 0)
#   trj, _ = sample_linear_trajectory(key_states, key_times, spatial(t=num_trj_frames))
#   #input_nn = math.rename_dims(trj_test.t[::2].elements.center, instance('balls'), spatial('balls'))
#   input_nn = math.rename_dims(trj_test.t[::6].elements.center, instance('balls'), spatial('balls'))
#   #input_nn_v = math.rename_dims(trj_test.values, instance('balls'), spatial('balls'))
#   input_speed = math.expand(v0_speed, shape(input_nn.vector[0]))
#   #input_alpha = math.expand(alpha, shape(input_nn.vector[0]))
#   input_nn_conv = math.stack([input_nn.vector[0], input_nn.vector[1], input_speed],channel(features='input_pos_x, input_pos_y, input_speed'))
#   #input_nn_conv = math.stack([input_nn.vector[0], input_nn.vector[1], input_alpha], channel(features='input_x, input_y, input_alpha'))
#   #input_nn_dense = math.flatten(trj.t[::6].values, channel('input'))
#   return math.l2_loss(trj.elements.center - trj_test.elements.center), math.mean(math.native_call(net, input_nn_conv)), math.mean(math.native_call(net2, input_nn_conv)),math.mean(math.native_call(net3, input_nn_conv))
#
# show({"Ground Truth": CenteredGrid(lambda v0_speed: landscape(v0_speed)[0], v0_speed=100, bounds=Box(v0_speed=(parm_1_min,parm_1_max)))})
# show({"ConvNET predicted": CenteredGrid(lambda v0_speed: landscape(v0_speed)[1], v0_speed=100, bounds=Box(v0_speed=(parm_1_min,parm_1_max)))})
# show({"ConvNET Noise 1": CenteredGrid(lambda v0_speed: landscape(v0_speed)[2], v0_speed=100, bounds=Box(v0_speed=(parm_1_min,parm_1_max)))})
# show({"ConvNET Noise 1 rf": CenteredGrid(lambda v0_speed: landscape(v0_speed)[3], v0_speed=100, bounds=Box(v0_speed=(parm_1_min,parm_1_max)))})
# # show({"ConvNET Noise 2": CenteredGrid(lambda alpha, v0_speed: landscape(v0_speed)[4], v0_speed=100, bounds=Box(v0_speed=(parm_1_min,parm_1_max)))})
# # show({"ConvNET Noise 2 rf": CenteredGrid(lambda alpha, v0_speed: landscape(v0_speed)[5], v0_speed=100, bounds=Box(v0_speed=(parm_1_min,parm_1_max)))})
#
#
#
#
# vis.show(CenteredGrid(lambda v0_speed: math.stack({"Ground Truth": landscape(v0_speed)[0],"ConvNet predicted": landscape(v0_speed)[1], "ConvNet noise1": landscape(v0_speed)[2], "ConvNet noise1_rf": landscape(v0_speed)[3]}, channel('curves')),v0_speed=100, bounds=Box(v0_speed = (parm_1_min,parm_1_max))))
# # vis.show(CenteredGrid(lambda alpha, v0_speed: math.stack({"Ground Truth": landscape(alpha, v0_speed)[0],"ConvNet predicted": landscape(alpha, v0_speed)[1], "ConvNet noise2": landscape(alpha, v0_speed)[4], "ConvNet noise2_rf": landscape(alpha, v0_speed)[5]}, channel('curves')), alpha=100,v0_speed=100, bounds=Box(alpha=(parm_1_min,parm_1_max), v0_speed = (parm_2_min,parm_2_max))))
# # vis.show(CenteredGrid(lambda alpha, v0_speed: math.stack({"Ground Truth": landscape(alpha, v0_speed)[0],"ConvNet predicted": landscape(alpha, v0_speed)[1], "ConvNet noise1": landscape(alpha, v0_speed)[2], "ConvNet noise2": landscape(alpha, v0_speed)[4]}, channel('curves')), alpha=100,v0_speed=100, bounds=Box(alpha=(parm_1_min,parm_1_max), v0_speed = (parm_2_min,parm_2_max))))
# vis.show(CenteredGrid(lambda alpha, v0_speed: math.stack({"Ground Truth": landscape(alpha, v0_speed)[0],"ConvNet predicted": landscape(alpha, v0_speed)[1], "ConvNet noise1_rf": landscape(alpha, v0_speed)[3], "ConvNet noise2_rf": landscape(alpha, v0_speed)[5]}, channel('curves')), alpha=100,v0_speed=100, bounds=Box(alpha=(parm_1_min,parm_1_max), v0_speed = (parm_2_min,parm_2_max))))