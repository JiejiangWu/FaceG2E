
def prompt_suffix(rotation):
    x=rotation[0];y=rotation[1];z=rotation[2]

    # 0-45, 315-360: front view
    # 135-225: back view
    # 45-135, 225-315: side view
    y = y % 360

    if (y >=0 and y <= 45) or (y>=315 and y <=360):
        return 'front view'
    if y > 135 and y < 225:
        return 'back view'
    if (y>45 and y <=135) or (y >=225 and y < 315):
        return 'side view'

def prompt_suffix_smpl(theta):
    theta = theta % 360

    if (theta >= 45 and theta <= 135):
        return 'front'
    if (theta >=0 and theta < 45) or (theta >=315 and theta < 360) or (theta>135 and theta<225):
        return 'side'
    if (theta >=225 and theta < 315):
        return 'back'

        