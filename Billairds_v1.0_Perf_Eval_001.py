# from phi.tf.flow import *
from phi.torch.flow import *
# from phi.jax.stax.flow import *
from phi._troubleshoot import plot_solves
import csv

#parameters
x0 = vec(x=0.4, y=0.6)
speed = 1
num_key_states = 15
num_trj_frames = 60
layers = 4 #billiards_layers
parm_1_min = -PI/5
parm_1_max =  PI/5
noise_scale_1= 0.14
noise_scale_2= 0.13
skew_scale_high= 3
skew_scale_low= 0.1

num_train_steps = 20000


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


net = conv_net(5, 1, [64,128,256,128,64], activation='ReLU', in_spatial=2)
optimizer = adam(net)
math.print(parameter_count(net))
net2 = conv_net(5, 1, [64,128,256,128,64], activation='ReLU', in_spatial=2)
optimizer2 = adam(net2)
math.print(parameter_count(net2))
net3 = conv_net(5, 1, [64,128,256,128,64], activation='ReLU', in_spatial=2)
optimizer3 = adam(net3)
net4 = conv_net(5, 1, [64,128,256,128,64], activation='ReLU', in_spatial=2)
optimizer4 = adam(net4)
net5 = conv_net(5, 1, [64,128,256,128,64], activation='ReLU', in_spatial=2)
optimizer5 = adam(net5)





load_state(net, './billiards_ConvNet_10042023_surrogate.pth')
load_state(net2, './billiards_ConvNet2_10042023_smooth_surrogate_n1.pth')
load_state(net3, './billiards_ConvNet3_10042023_forced_surrogate_n1.pth')
# load_state(net4, './billiards_ConvNet4_10042023_fs_noise2.pth')
# load_state(net5, './billiards_ConvNet5_10042023_rf.pth')


min_low = -PI/6
min_high = PI/6

alpha_true = math.random_uniform(low=min_low, high=min_high)
math.print(alpha_true)


balls = billiards_triangle()
cue_ball = Sphere(tensor([x0], instance('balls'), channel(vector='x,y')), radius=.03)
all_balls = math.concat([cue_ball, balls], instance('balls'))
balls_v = PointCloud(balls, tensor([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)], shape(balls)))
v0_speed = speed
v0_true = vec(x=v0_speed * math.cos(alpha_true), y=v0_speed * math.sin(alpha_true))
cue_ball_v_true = PointCloud(cue_ball, tensor([(v0_true)], shape(cue_ball)))
all_balls_v_true = math.concat([cue_ball_v_true, balls_v], instance('balls'))

key_states_true, key_times_true = iterate(physics_step, spatial(keys=num_key_states), all_balls_v_true, 0)
trj_true, _ = sample_linear_trajectory(key_states_true, key_times_true, spatial(t=num_trj_frames))



def plot_fn(alpha):
  v0_test = vec(x=v0_speed * math.cos(alpha), y=v0_speed * math.sin(alpha))
  cue_ball_v_test = PointCloud(cue_ball, tensor([(v0_test)], shape(cue_ball)))
  all_balls_v_test = math.concat([cue_ball_v_test, balls_v], instance('balls'))
  key_states_test, key_times_test = iterate(physics_step, spatial(keys=num_key_states), all_balls_v_test, 0)
  trj_test, _ = sample_linear_trajectory(key_states_test, key_times_test, spatial(t=num_trj_frames))

  input_net = math.rename_dims(trj_true.t[::2].elements.center, instance('balls'), spatial('balls'))
  input_net_v = math.rename_dims(trj_true.t[::2].values, instance('balls'), spatial('balls'))
  input_alpha = math.expand(alpha, shape(input_net.vector[0]))
  input_net_conv = math.stack([input_net.vector[0], input_net_v.vector[0], input_net.vector[1], input_net_v.vector[1], input_alpha],channel(features='input_pos_x,input_vel_x, input_pos_y,input_vel_y, input_alpha'))

  return math.l2_loss(trj_test.elements.center - trj_true.elements.center), math.mean(math.native_call(net, input_net_conv)), math.mean(math.native_call(net2, input_net_conv)), math.mean(math.native_call(net3, input_net_conv))



show(CenteredGrid(lambda alpha: math.stack({"Ground Truth Configuration loss": plot_fn(alpha)[0],"Surrogate predicted (S)": plot_fn(alpha)[1], "Biased Smooth Surrogate (BSS) predicted": plot_fn(alpha)[3]}, channel('curves')), alpha=100, bounds=Box(alpha=(parm_1_min, parm_1_max))), size=(15,7), title='Network Outputs')


# alpha_true = math.random_uniform(low=min_low, high=min_high)
# math.print(alpha_true)


v0_true = vec(x=v0_speed * math.cos(alpha_true), y=v0_speed * math.sin(alpha_true))
cue_ball_v_true = PointCloud(cue_ball, tensor([(v0_true)], shape(cue_ball)))
all_balls_v_true = math.concat([cue_ball_v_true, balls_v], instance('balls'))

key_states_true, key_times_true = iterate(physics_step, spatial(keys=num_key_states), all_balls_v_true, 0)
trj_true, _ = sample_linear_trajectory(key_states_true, key_times_true, spatial(t=num_trj_frames))

def gt_min(x0):
    with math.SolveTape() as tape:
      #min_gt = math.minimize(lambda alpha: plot_fn(alpha)[0], Solve('BFGS', 0, 1e-5, x0=x0, suppress=[Diverged, NotConverged]))
      min_gt = math.minimize(lambda alpha: plot_fn(alpha)[0], Solve('BFGS', 0, 1e-5, x0=x0, suppress=[Diverged, NotConverged]))
      return min_gt
        #return(fn(sol))
        #return sol - x0
        #return tape.solves[0].iterations

def surrogate_min(x0):
    with math.SolveTape() as tape:
      #min_gt = math.minimize(lambda alpha: plot_fn(alpha)[0], Solve('BFGS', 0, 1e-5, x0=x0, suppress=[Diverged, NotConverged]))
      min_surrogate = math.minimize(lambda alpha: plot_fn(alpha)[1], Solve('BFGS', 0, 1e-5, x0=x0, suppress=[Diverged, NotConverged]))
      return min_surrogate
        #return(fn(sol))
        #return sol - x0
        #return tape.solves[0].iterations

def primary_min(x0):
    with math.SolveTape() as tape:
      min_forced_surrogate = math.minimize(lambda alpha: plot_fn(alpha)[3], Solve('BFGS', 0, 1e-5, x0=x0, suppress=[Diverged, NotConverged]))
      #return min_gt, min_surrogate, min_smooth_surrogate, min_forced_surrogate
      return min_forced_surrogate
        #return(fn(sol))
        #return sol - x0
        #return tape.solves[0].iterations

def secondary_min(x0):
    with math.SolveTape() as tape:
      min_sec_fs_to_s = math.minimize(lambda alpha: plot_fn(alpha)[1], Solve('BFGS', 0, 1e-5, x0=primary_min(x0), suppress=[Diverged, NotConverged]))
      #min_sec_ss_to_s = math.minimize(lambda alpha: plot_fn(alpha)[1], Solve('BFGS', 0, 1e-5, x0=min_sec_fs_to_ss, suppress=[Diverged, NotConverged]))
      return min_sec_fs_to_s
        #return(fn(sol))
        #return sol - x0
        #return tape.solves[0].iterations


#show(CenteredGrid(lambda alpha_test: math.stack({"Ground Truth": alpha_true, "Surrogate Optimization": surrogate_min(alpha_test),"Primary RS Optimisation": primary_min(alpha_test),"Final Optimization": secondary_min(alpha_test)}, channel('curves')), alpha_test=100, bounds=Box(alpha_test=(parm_1_min, parm_1_max))), size=(12,5), title='Optimization Performance')


alpha_true_list=[]
alpha_check_list=[]
primary_min_list=[]
secondary_min_list=[]
delta_alpha_list=[]
resim_error_list=[]

batch_size = 128
num_evals = 8
num_checks = 10
for evals in range(num_evals):
  alpha_true = math.random_uniform(batch(b=batch_size), low=min_low, high=min_high) #True value of the control parameter to be detected
  print(f"alpha_true: {alpha_true}")
  v0_true = vec(x=v0_speed * math.cos(alpha_true), y=v0_speed * math.sin(alpha_true))
  cue_ball_v_true = PointCloud(cue_ball, tensor([(v0_true)], shape(cue_ball)))
  all_balls_v_true = math.concat([cue_ball_v_true, balls_v], instance('balls'))
  key_states_true, key_times_true = iterate(physics_step, spatial(keys=num_key_states), all_balls_v_true, 0)
  trj_true, _ = sample_linear_trajectory(key_states_true, key_times_true, spatial(t=num_trj_frames))
  for checks in range(num_checks):
    alpha_check = math.random_uniform(batch(b=batch_size),low=min_low, high=min_high) #Starting value of the control parameter in the minimisation corresponding to 1 true control parameter
    print(f"alpha_check: {alpha_check}")
    primary_minima = primary_min(alpha_check)
    secondary_minima = secondary_min(alpha_check)
    delta_alpha = math.l1_loss(secondary_minima - alpha_true)
    alpha_true_final = math.rename_dims(alpha_true, batch, instance)
    alpha_check_final = math.rename_dims(alpha_check, batch, instance)
    primary_minima_final = math.rename_dims(primary_minima, batch, instance)
    secondary_minima_final = math.rename_dims(secondary_minima, batch, instance)
    v0_resim = vec(x=v0_speed * math.cos(secondary_minima), y=v0_speed * math.sin(secondary_minima))
    cue_ball_v_resim = PointCloud(cue_ball, tensor([(v0_resim)], shape(cue_ball)))
    all_balls_v_resim = math.concat([cue_ball_v_resim, balls_v], instance('balls'))
    key_states_resim, key_times_resim = iterate(physics_step, spatial(keys=num_key_states), all_balls_v_resim, 0)
    trj_resim, _ = sample_linear_trajectory(key_states_resim, key_times_resim, spatial(t=num_trj_frames))
    resim_error = math.l2_loss(trj_resim.elements.center - trj_true.elements.center)
    delta_alpha_final = math.rename_dims(delta_alpha, batch, instance)
    resim_error_final = math.rename_dims(resim_error, batch, instance)
    alpha_true_list.append(alpha_true_final)
    alpha_check_list.append(alpha_check_final)
    primary_min_list.append(primary_minima_final)
    secondary_min_list.append(secondary_minima_final)
    delta_alpha_list.append(delta_alpha_final)
    resim_error_list.append(resim_error_final)
    #tertiary_minima = tertiary_min(alpha_check)
    #primary_min_error = (math.l1_loss(primary_minima - alpha_true)*100)
    #secondary_min_error = (math.l1_loss(secondary_minima - alpha_true)*100)
    #tertiary_min_error = (math.l1_loss(tertiary_minima - alpha_true)*100)
    print(f"Evaluation_parameters:{alpha_true, alpha_check, primary_minima,secondary_minima, delta_alpha, resim_error}")




primary_perf_scatter = stack(alpha_true_list + primary_min_list, math.concat_shapes(channel(vector='True angle of strike , Primary Step Convergence'), instance(points=num_evals*num_checks)))
vis.show(primary_perf_scatter)
math.print(primary_perf_scatter)

secondary_perf_scatter = stack(alpha_true_list + secondary_min_list, math.concat_shapes(channel(vector='True angle of strike , Angle of strike predicted'), instance(points=num_evals*num_checks)))
vis.show(secondary_perf_scatter)
math.print(secondary_perf_scatter)

delta_perf_scatter = stack(alpha_true_list + delta_alpha_list, math.concat_shapes(channel(vector='True angle of strike, Prediction_Error'), instance(points=num_evals*num_checks)))
vis.show(delta_perf_scatter)
math.print(delta_perf_scatter)

resim_perf_scatter = stack(alpha_true_list + resim_error_list, math.concat_shapes(channel(vector='True angle of strike, Resimulation_Error'), instance(points=num_evals*num_checks)))
vis.show(resim_perf_scatter)
math.print(resim_perf_scatter)

hist, edges, centers = math.histogram(delta_perf_scatter['Prediction_Error'], bins=spatial(Prediction_Error=30))
show(delta_perf_scatter, PointCloud(centers, hist))

hist, edges, centers = math.histogram(delta_perf_scatter['Prediction_Error'], bins=spatial(Prediction_Error=30))
show(delta_perf_scatter, PointCloud(centers, hist),  same_scale='Prediction_Error', log_dims='_')

hist, edges, centers = math.histogram(delta_perf_scatter['Prediction_Error'], bins=spatial(Prediction_Error=30))
show(delta_perf_scatter, PointCloud(centers, hist),  same_scale='True angle of strike', log_dims='_')

hist, edges, centers = math.histogram(resim_perf_scatter['Resimulation_Error'], bins=spatial(Resimulation_Error=30))
show(resim_perf_scatter, PointCloud(centers, hist))

hist, edges, centers = math.histogram(resim_perf_scatter['Resimulation_Error'], bins=spatial(Resimulation_Error=30))
show(resim_perf_scatter, PointCloud(centers, hist),  same_scale='Resimulation_Error', log_dims='_')

hist, edges, centers = math.histogram(resim_perf_scatter['Resimulation_Error'], bins=spatial(Resimulation_Error=30))
show(resim_perf_scatter, PointCloud(centers, hist),  same_scale='True angle of strike', log_dims='_')


with open('alpha_true', 'w') as f:
    write = csv.writer(f)
    write.writerow(alpha_true_list)
with open('alpha_check', 'w') as f:
    write = csv.writer(f)
    write.writerow(alpha_check_list)
with open('primary_min_list', 'w') as f:
    write = csv.writer(f)
    write.writerow(primary_min_list)
with open('secondary_min_list', 'w') as f:
    write = csv.writer(f)
    write.writerow(secondary_min_list)
with open('delta_alpha_list', 'w') as f:
    write = csv.writer(f)
    write.writerow(delta_alpha_list)