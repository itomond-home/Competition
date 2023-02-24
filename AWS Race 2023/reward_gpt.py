import math
def reward_function(params):
    # パラメータ読み込み
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    steering = abs(params['steering_angle']) 
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    progress = params['progress']

    # 初期報酬
    reward = 1.0

    # コースの中心から左にどの程度ずれているかを計算
    left_lane_pos = track_width / 4.0
    if distance_from_center <= left_lane_pos:
        reward *= 1.0
    else:
        reward *= 0.1

    # 車両がコース内にあることを確認
    if not all_wheels_on_track:
        reward *= 0.0001
        
    # 速度が遅い場合は報酬を減らす
    if speed < 1.0:
        reward *= 0.5

    # 進捗度合いに応じて報酬を追加
    reward += progress / 100

    # waypointによる報酬の設定
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    track_direction = math.degrees(track_direction)

    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff

    # カーブを曲がりきれた場合に報酬を追加
    if direction_diff < 10:
        reward += 1.0
    elif direction_diff < 20:
        reward += 0.5
    else:
        reward += 0.01

    if abs(direction_diff) < 5 and abs(steering) < 5 and speed > 3:
        # 直進状態であれば報酬を加算
        reward += speed / 5

    # 車がトラック上にあるかどうか
    if not all_wheels_on_track:
        reward = 1e-3

    return reward
