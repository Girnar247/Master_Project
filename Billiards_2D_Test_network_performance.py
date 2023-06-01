# from phi.tf.flow import *
from phi.torch.flow import *
# from phi.jax.stax.flow import *
from phi._troubleshoot import plot_solves

#parameters
x0 = vec(x=0.4, y=0.6)
speed = 1
num_key_states = 15
num_trj_frames = 60
layers = 4 #billiards_layers
num_input_dense = 2*(num_key_states + 1)*((1 + 2 + 3 + 4) + 1)
fixed_goal = vec(x=2.4, y=0.5)
parm_1_min = -PI/6
parm_1_max =  PI/6
parm_2_min = 0.8
parm_2_max =  2
noise_scale_1= 0.18
noise_scale_2= 0.28
skew_scale_high= 3
skew_scale_low= 0.1
num_train_steps = 5000

def billiards_triangle(billiard_layers=layers, radius=.03):
    coords = []
    for i in range(billiard_layers):
        for j in range(i + 1):
            coords.append(vec(x=i * 2.4 * radius + 0.6, y=j * 2.4 * radius + 0.6 - i * radius * 0.7))
    return Sphere(stack(coords, instance('balls')), radius=radius)


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

net = conv_net(4, 1, [64,128,128,64], activation='ReLU', in_spatial=2)
optimizer = adam(net)
math.print(parameter_count(net))
net2 = conv_net(4, 1, [64,128,128,64], activation='ReLU', in_spatial=2)
optimizer2 = adam(net2)
math.print(parameter_count(net2))
net3 = conv_net(4, 1, [64,128,128,64], activation='ReLU', in_spatial=2)
optimizer3 = adam(net3)
net4 = conv_net(4, 1, [64,128,128,64], activation='ReLU', in_spatial=2)
optimizer4 = adam(net4)
net5 = conv_net(4, 1, [64,128,128,64], activation='ReLU', in_spatial=2)
optimizer5 = adam(net5)

load_state(net, './billiards_ConvNet_2.1_22032023_01.pth')
load_state(net2, './billiards_ConvNet_2.1_22032023_noise_1_01.pth')
load_state(net3, './billiards_ConvNet_2.1_22032023_noise_1_rf_01.pth')
load_state(net4, './billiards_ConvNet_2.1_22032023_noise_2_01.pth')
load_state(net5, './billiards_ConvNet_2.1_22032023_noise_2_rf_01.pth')

alpha_test = math.random_uniform( low=parm_1_min, high=parm_1_max)
math.print(alpha_test)

v0_speed_test = math.random_uniform( low=parm_2_min, high=parm_2_max)
math.print(v0_speed_test)

def landscape(alpha, v0_speed):
  balls = billiards_triangle()
  x0_test = x0
  goal = fixed_goal
  cue_ball = Sphere(tensor([x0_test], instance('balls'), channel(vector='x,y')), radius=.03)
  all_balls = math.concat([cue_ball, balls], instance('balls'))
  #balls_v = PointCloud(balls, tensor([(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)], shape(balls)))
  balls_v = PointCloud(balls, tensor([(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)], shape(balls)))
  #v0_speed = speed
  # math.print(alpha_test)
  # math.print(v0_speed)
  v0 = vec(x=v0_speed * math.cos(alpha), y=v0_speed * math.sin(alpha))
  v0_test = vec(x=v0_speed_test * math.cos(alpha_test), y=v0_speed_test * math.sin(alpha_test))
  cue_ball_v = PointCloud(cue_ball, tensor([(v0)], shape(cue_ball)))
  cue_ball_v_test = PointCloud(cue_ball, tensor([(v0_test)], shape(cue_ball)))
  all_balls_v = math.concat([cue_ball_v, balls_v], instance('balls'))
  all_balls_v_test = math.concat([cue_ball_v_test, balls_v], instance('balls'))
  key_states, key_times = iterate(physics_step, spatial(keys=num_key_states), all_balls_v, 0)
  key_states_test, key_times_test = iterate(physics_step, spatial(keys=num_key_states), all_balls_v_test, 0)
  trj, _ = sample_linear_trajectory(key_states, key_times, spatial(t=num_trj_frames))
  trj_test, _ = sample_linear_trajectory(key_states_test, key_times_test, spatial(t=num_trj_frames))
  #input_nn = math.rename_dims(trj_test.t[::2].elements.center, instance('balls'), spatial('balls'))
  input_nn = math.rename_dims(trj_test.t[::6].elements.center, instance('balls'), spatial('balls'))
  #input_nn_v = math.rename_dims(trj_test.values, instance('balls'), spatial('balls'))
  input_speed = math.expand(v0_speed, shape(input_nn.vector[0]))
  input_alpha = math.expand(alpha, shape(input_nn.vector[0]))
  input_nn_conv = math.stack([input_nn.vector[0], input_nn.vector[1], input_alpha, input_speed],channel(features='input_pos_x, input_pos_y, input_alpha, input_speed'))
  #input_nn_conv = math.stack([input_nn.vector[0], input_nn.vector[1], input_alpha], channel(features='input_x, input_y, input_alpha'))
  #input_nn_dense = math.flatten(trj.t[::6].values, channel('input'))
  #return math.l2_loss(trj.elements.center - trj_test.elements.center), math.mean(math.native_call(net, input_nn_conv)), math.mean(math.native_call(net2, input_nn_conv)),math.mean(math.native_call(net3, input_nn_conv)),math.mean(math.native_call(net4, input_nn_conv)), math.mean(math.native_call(net5, input_nn_conv))
  return math.l2_loss(trj.elements.center - trj_test.elements.center)

show({"Ground Truth": CenteredGrid(lambda alpha, v0_speed: landscape(alpha, v0_speed)[0], alpha=50,v0_speed=50, bounds=Box(alpha=(parm_1_min,parm_1_max), v0_speed=(parm_2_min,parm_2_max)))})
# show({"ConvNET": CenteredGrid(lambda alpha, v0_speed: landscape(alpha, v0_speed)[1], alpha=50,v0_speed=50, bounds=Box(alpha=(parm_1_min,parm_1_max), v0_speed=(parm_2_min,parm_2_max)))})
# show({"ConvNET Noise1": CenteredGrid(lambda alpha, v0_speed: landscape(alpha, v0_speed)[2], alpha=50,v0_speed=50, bounds=Box(alpha=(parm_1_min,parm_1_max), v0_speed=(parm_2_min,parm_2_max)))})
# show({"ConvNET Noise1 Rf": CenteredGrid(lambda alpha, v0_speed: landscape(alpha, v0_speed)[3], alpha=50,v0_speed=50, bounds=Box(alpha=(parm_1_min,parm_1_max), v0_speed=(parm_2_min,parm_2_max)))})
# show({"ConvNET Noise2": CenteredGrid(lambda alpha, v0_speed: landscape(alpha, v0_speed)[4], alpha=50,v0_speed=50, bounds=Box(alpha=(parm_1_min,parm_1_max), v0_speed=(parm_2_min,parm_2_max)))})
# show({"ConvNET Noise 2 Rf": CenteredGrid(lambda alpha, v0_speed: landscape(alpha, v0_speed)[5], alpha=50,v0_speed=50, bounds=Box(alpha=(parm_1_min,parm_1_max), v0_speed=(parm_2_min,parm_2_max)))})


def find_min_truth(x0):
  with math.SolveTape() as tape:
    sol = math.minimize(landscape, Solve('BFGS', 0, 1e-5, x0=x0, suppress=[Diverged, NotConverged]))
    # return sol
    # return fn(sol)
    return sol - x0
    #return tape.solves[0].iterations


iterations_gt = CenteredGrid(find_min_truth, alpha=10, v0_speed=10, bounds=Box(alpha=(-PI/6,PI/6), v0_speed=(0.5,1)))

vis.show(iterations_gt)