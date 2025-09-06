import os
import json
import math

def write_stats_file(path, filename, name, render_p, track_p,error, cancelled = False):
    progress = {'name' : name,
                  'tracking_progress' : track_p,
                  'render_progress' : render_p, 
                  'filename' : filename,
                  'is_cancelled':cancelled,
                  'error' : error}
    with open(os.path.join(path, f'{name}_render_progress.json').replace('\\','/'), 'w') as f:
        json.dump(progress, f, indent = 4)

def calc_progress(boxes_filt, id, step, total_step):
        #TODO This needs fixing as it can go above 100%
        value = math.ceil(((step * id) / (total_step * boxes_filt)) * 100)
        return f'{value}%'