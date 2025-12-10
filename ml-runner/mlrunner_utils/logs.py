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
        total_s = total_step * boxes_filt
        current = id * total_step + step
        value = int((current/total_s) * 100)
        return f'{value}%'

def check_for_abort_render(render_dir, shot_name, uuid, logger, is_tracking = False):
    is_abort = os.path.isfile(os.path.join(render_dir,'cancel_render').replace('\\','/'))
    if is_abort and not is_tracking:
        os.remove(os.path.join(render_dir,'cancel_render').replace('\\','/'))
    if is_abort:
        logger.info('Interrupting render as we detected an abort request from the user')
        write_stats_file(render_dir, shot_name, uuid, '0%', '0%', 'False', 'True')
    return is_abort

