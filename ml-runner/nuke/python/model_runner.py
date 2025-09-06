import json
import os
import uuid
import nuke
import nukescripts

def ensure_legal_crop_size(crop_val, width, height):
    box = []
    for idx, i in enumerate(crop_val):
        if i <0: 
            i = 0
        elif i > width and idx == 2:
            i = width
        elif i > height and idx == 3:
            i = height
        box.append(i)
    return box

def ensure_legal_frame(current_frame, first, last):
    frame = current_frame if current_frame > first else first
    frame = current_frame if current_frame < last else last 
    return frame

def setup_read(file_list, node):
    """Function to create the read node. Have to use this as a workaround as I cant simply do nuke.nodes.Read() with threading.Timer as it crashes nuke"""

    reads = [i for i in nuke.allNodes('Read')]
    read_no = max([int(i['name'].value().replace('Read','')) for i in reads if 'MLRunner' not in i.name()]) + 1
    template_node = """Read {
        name template_name
        file template_file
        first template_first
        last template_last
        file_type png
        xpos template_xpos
        ypos template_ypos
        }"""
    name, frame_range = file_list.split(' ')
    first,last = frame_range.split('-')
    xpos = node.xpos() + 150
    ypos = node.ypos()
    template_node = template_node.replace('template_file',name).replace('template_first', first).replace('template_last', last).replace('template_name','Read' + str(read_no)).replace('template_xpos', str(xpos)).replace('template_ypos', str(ypos))
    nuke.tcl('in root {%s}' % template_node)


def load_bg_render(node, path, print_to_terminal = True):
    """Function to check every n seconds the render progress. When progress reaches 100% it loads the read node into nuke"""
    timer = threading.Timer(10, load_bg_render, [node, path],)
    timer.start()

    if os.path.isfile(path):
        # Read json
        with open(path,'r') as f:
            file = json.load(f)
        name = file['name']
        tracking = file['tracking_progress']
        render_p = file['render_progress']
        filename = file['filename']
        error = file['error']
        cancelled = file['is_cancelled']

        # Check if file is the first in queue
        queue = node['queue'].value().split(',')
        queue = queue if len(queue) >= 1 else [name]
        correct_queue = True if name == queue[0] else False
        timer_completed = False

        # Update progress
        if correct_queue:
            # This is a little buggy as nuke UI doens't always update 
            filler = f'name: {name}\ntracking progress: {tracking}\nrender_progress: {render_p}'
            node['render_progress'].setValue(filler)
            if print_to_terminal:
                nuke.tprint(filler)

        # Stop timer
        if render_p == '100%' or error or cancelled:
            timer.cancel()
            queue = node['queue'].value().split(',')
            queue.pop(0)
            node['queue'].setValue(','.join(i for i in queue))
            timer_completed = True
            os.remove(path)
            # Need to implement error behaviour

        # Load rendered element in nuke
        if timer_completed and not error and not cancelled:
            shot = [i for i in nuke.getFileNameList(os.path.dirname(path)) if filename in i]
            if shot:
                setup_read(os.path.join(os.path.dirname(path),shot[0]).replace('\\','/'), node)

def padding_to_num(padded_name, delimiter, ref_frame):
    pad, ext = os.path.splitext(padded_name)
    pad = pad.split(delimiter)[-1]
    pad = pad.replace('%0','').replace('d','')
    pad = int(pad) - len(str(ref_frame))
    pads = ''.join('0' for i in range(int(pad)))
    return pads

node = nuke.thisNode()
config_save_directory = node['config_save_directory'].value()
shot_name = node['shot_name'].value()
width = node.width()
height = node.height()
delimiter = node['delimiter'].value()
frame_idx = int(node['frame_idx'].value()) #ensure_legal_frame(int(node['frame_idx'].value()), int(node.firstFrame()), int(node.lastFrame()))

# Check all is good
is_correct_extension = any(shot_name.split('.')[-1].lower() in i for i in ('.png','.jpg', '.jpeg', '.exr'))
path_to_sequence_exists = os.path.isdir(node['path_to_sequence'].value())
is_node_connected = len(node.dependencies()) > 0
is_frame_legal = frame_idx >= int(node.firstFrame()) and frame_idx <= int(node.lastFrame())
if node['use_limit'].value():
    is_frame_legal = frame_idx >= int(node['limit_first'].value()) and frame_idx <= int(node['limit_last'].value())
config_dir_exists = os.path.isdir(config_save_directory) 
is_server_running = os.path.isfile(os.path.join(config_save_directory,'.server_is_running.tmp').replace('\\','/')) 

all_good = False
if all(i for i in (is_correct_extension, path_to_sequence_exists, is_node_connected, is_frame_legal, config_dir_exists, is_server_running)):
    all_good = True
else:
    errors = [idx for idx,i in enumerate((is_correct_extension, path_to_sequence_exists, is_node_connected, is_frame_legal, config_dir_exists, is_server_running)) if not i]
    error_keys = ['unsupported_extension', 'wrong_path', 'node_not_connected', 'wrong_frame', 'wrong_config_path', 'server_not_running']

error_messages = {
    'unsupported_extension' : 'ERROR! The sequence you are trying to run does not have the correct file extension. We only support png and jpg for now.\n',
    'wrong_path'            : 'ERROR! The path to the sequence doesnt seem to exists. Please check for spelling mistakes or use the Fetch Path shot button\n',
    'node_not_connected'    : 'ERROR! Please connect the node to the sequence you want to run\n',
    'wrong_frame'           : 'ERROR! The reference frame is outside the range of the sequence\n',
    'wrong_config_path'     : f"ERROR! The path defined in the Server listenting directory knob doesn't seem to exist! Please check for spelling mistakes!\n {config_save_directory}\n",
    'server_not_running'    : f"ERROR! The server does't seem to be running. \nPlease contact your administrator or ensure that the server is listenting in the same directory as what's specified in the node:\n{config_save_directory}\n",
}

if all_good:
    crop_position = ensure_legal_crop_size(node['crop_position'].value(), width, height)
    ## Setup config file
    config = {}
    config['listen_dir']   = config_save_directory
    config['model_to_run'] = node['model_to_run'].value()
    config['id_class'] = node['id_class'].value() 
    config['crop_position'] = [crop_position]
    config['path_to_sequence'] = node['path_to_sequence'].value()
    config['shot_name'] = shot_name
    config['render_to'] = node['render_to'].value()
    config['render_name'] = node['render_name'].value()
    config['use_gdino'] = node['use_gdino'].value()
    config['frame_idx'] = int(frame_idx - node.firstFrame())
    config['first_frame_sequence'] = int(node.firstFrame())
    config['limit_range'] = node['use_limit'].value()
    config['delimiter'] = delimiter

    # Not implemented yet
    config['multisequence'] = False

    # We remove one frame as we are using these to simply slice lists
    config['limit_first'] = int(node['limit_first'].value() - node.firstFrame())
    config['limit_last'] = int(node['limit_last'].value() - node.firstFrame()) + 1

    # Write related stuff
    write_sequence = node['write_sequence'].value()
    write = nuke.toNode(f'{node.name()}.Write1')

    # Workaround to evaluate padding
    write['file'].setValue(shot_name)
    shot_name_no_padding = os.path.basename(write['file'].evaluate())

    ## When using pipelines evaluate is returning the path of the sequence. This is a hacky workaround.
    if os.path.isdir(shot_name_no_padding):
        shot_name_no_padding = padding_to_num(shot_name, delimiter, int(frame_idx))
    config['shot_name'] = shot_name_no_padding

    if not os.path.isdir(config['render_to']):
        os.makedirs(config['render_to'])

    # If render sequence we write to path_to_sequence
    if write_sequence:
        write['file'].setValue(os.path.join(node['path_to_sequence'].value(), shot_name).replace('\\','/'))
        nukescripts.render_panel((nuke.thisNode(),), False)

    ## Write config
    unique_name = str(uuid.uuid4())
    config['uuid'] = unique_name
    with open(os.path.join(config_save_directory, f'{unique_name}_{shot_name_no_padding}.json').replace('\\','/'), 'w') as f:
        json.dump(config, f, indent = 4)

    queue = node['queue'].value()
    queue = queue + ',' + unique_name if queue != '' else unique_name
    node['queue'].setValue(queue)

    load_bg_render(node, os.path.join(config['render_to'], f'{unique_name}_render_progress.json').replace('\\','/'))
else:
    error_message = '\n'.join(error_messages[error_keys[i]] for i in errors )
    nuke.alert(f'The following errors where found. Please ensure they are all fixed prior to writing the config:\n{error_message}')
