import math
def reward_function(params):
    # パラメータ読み込み
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    abs_steering = abs(params['steering_angle']) 
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    objects_left_of_center = params['objects_left_of_center']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    
    # 報酬の初期化
    reward = 1.0    

    ## センターラインからの距離
    # センターラインからの距離を３分割
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # センターラインから右側ほど点数が低い
    if not objects_left_of_center:
        if distance_from_center <= marker_1:
            reward = 1.0
        elif distance_from_center <= marker_2:
            reward *= 0.5
        elif distance_from_center <= marker_3:
            reward *= 0.1
        else:
            reward *= 1e-3 # likely crashed/ close to off track

    ## 車輪の角度
    # ジグザグ走行を防ぐ
    ABS_STEERING_THRESHOLD = 20.0
    if abs_steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8

    ## タイヤが道路外に出ないようにする
    # 速度の閾値を設定
    SPEED_THRESHOLD = 1.0

    if not all_wheels_on_track:
        # Penalize if the car goes off track
        reward *= 1e-3
    elif speed < SPEED_THRESHOLD:
        # Penalize if the car goes too slow
        reward *= 0.1
    
    ## 車の方向が道路に沿っているか
    # 近場のウェイポイントから道路の角度を推定
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    track_direction = math.degrees(track_direction)

    # 車の方向と道路の方向の違いを算出
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    # 方向が違いすぎる場合は報酬を減らす
    DIRECTION_THRESHOLD = 10.0
    if direction_diff > DIRECTION_THRESHOLD:
        reward *= 0.01

    return float(reward)