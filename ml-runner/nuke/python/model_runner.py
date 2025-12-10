import json
import os
import uuid
import nuke
import nukescripts
import time


if nuke.NUKE_VERSION_MAJOR > 15:
    from PySide6 import QtWidgets, QtGui
    # from PySide6.QtWidgets import QPushButton, QSizePolicy
    from PySide6.QtCore import Qt, QObject, QEvent
else:
    from PySide2 import QtWidgets, QtGui
    from PySide2.QtCore import Qt, QObject, QEvent


def cancel_task():
    config_save_directory = node['render_to'].value()
    with open(os.path.join(config_save_directory, 'cancel_render'),'w') as f:
        f.write('cancel')

def run_progress_bar(path_to_file, model_type):
    task = nuke.ProgressTask("Please wait...")

    spinner = ["-", "\\", "|", "/"]
    i = 0

    in_progress = True
    filename = None
    render_progress = 0
    progress = 0
    tracking_progress = 0
    while in_progress:

        if task.isCancelled():
            nuke.message("Cancelled!")
            task.setProgress(100)
            cancel_task()
            return False

        if os.path.isfile(path_to_file):
            try:
                with open(path_to_file,'r') as f:
                    cfg = json.load(f)
                
                render_progress = cfg['render_progress']
                tracking_progress = cfg['tracking_progress']
                filename = cfg['filename']
            except:
                pass

            progress = int((int(render_progress.replace('%','')) + int(tracking_progress.replace('%',''))) / 2 )

        task.setProgress((progress % 100))
        task.setMessage(f"Processing {model_type} {spinner[i % 4]}")

        time.sleep(0.2)

        i += 1
        in_progress = render_progress != "100%"
        if not in_progress:
            break
    task.setProgress(100)

    return filename

def remove_from_queue(node, path):
    queue = node['queue'].value().split(',')
    queue.pop(0)
    node['queue'].setValue(','.join(i for i in queue))
    if os.path.isfile(path):
        os.remove(path)

class UserActivityMonitor(QObject):
    def __init__(self, idle_timeout=1):  # milliseconds
        super().__init__()
        self.last_activity_time = time.time()
        self.idle_timeout = idle_timeout / 1000  # convert to seconds


    def eventFilter(self, obj, event):
        if event.type() in (QEvent.KeyPress, QEvent.MouseMove, QEvent.MouseButtonPress):
            self.last_activity_time = time.time()
        return super().eventFilter(obj, event)


    def is_user_idle(self):
        return (time.time() - self.last_activity_time) > self.idle_timeout


class CustomTimer(threading.Thread):
    def __init__(self, interval, function, args=None, kwargs=None):
        super().__init__()
        self.interval = interval
        self.function = function
        self.args = args or []
        self.kwargs = kwargs or {}
        self._stop_event = threading.Event()
        self.has_created = False


    def run(self):
        if not self._stop_event.wait(self.interval):
            self.has_created = self.function(*self.args, **self.kwargs)
        if self.has_created:
            self.cancel()


    def cancel(self):
        self._stop_event.set()


class Loader(object):
    def __init__(self, node, path, print_to_terminal = True, is_tracker = False):
        self.node = node
        self.path = path
        self.print_to_terminal = print_to_terminal
        self.is_tracker = is_tracker
        self.timer = CustomTimer(10, self.load_bg_render, [self.node, self.path, self.print_to_terminal, self.is_tracker],)
        self.timer.start()
        self.timer_completed = False
        self.user_monitor = UserActivityMonitor()
        app = QtWidgets.QApplication.instance()
        app.installEventFilter(self.user_monitor)


    def setup_read(self, file_list, node):
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
        if ' ' in file_list:
            name, frame_range = file_list.split(' ')
            first,last = frame_range.split('-')
            xpos = node.xpos() + 150
            ypos = node.ypos()
            template_node = template_node.replace('template_file',name).replace('template_first', first).replace('template_last', last).replace('template_name','Read' + str(read_no)).replace('template_xpos', str(xpos)).replace('template_ypos', str(ypos))
            nuke.tcl('in root {%s}' % template_node)
            time.sleep(0.2)

    def load_bg_render(self, node, path, print_to_terminal, is_tracker):
        if not self.timer_completed:
            self.timer = CustomTimer(10, self.load_bg_render, [self.node, self.path, self.print_to_terminal, self.is_tracker],)
            self.timer.start()
        else:
            self.timer.cancel()


        if os.path.isfile(self.path):
            # Read json
            with open(self.path,'r') as f:
                file = json.load(f)
            name = file['name']
            tracking = file['tracking_progress']
            render_p = file['render_progress']
            filename = file['filename']
            error = file['error']
            cancelled = file['is_cancelled']


            # Check if file is the first in queue
            queue = self.node['queue'].value().split(',')
            queue = queue if len(queue) >= 1 else [name]
            correct_queue = True if name == queue[0] else False
            self.timer_completed = False


            # Update progress
            if correct_queue:
                # This is a little buggy as nuke UI doens't always update 
                filler = f'name: {name}\ntracking progress: {tracking}\nrender_progress: {render_p}'
                self.node['render_progress'].setValue(filler)
                if self.print_to_terminal:
                    nuke.tprint(filler)


            # Stop timer
            if render_p == '100%' or error or cancelled:
                remove_from_queue(self.node, self.path)
                self.timer_completed = True
                # Need to implement error behaviour


            # Load rendered element in nuke
            if self.timer_completed and not error and not cancelled:
                # Need to check if user is doing stuf as it can crash nuke if we create nodes while user interacts with UI
                max_count = 40
                count = 0
                while count <= max_count:
                    if self.user_monitor.is_user_idle():
                        if not self.is_tracker:
                            if self.user_monitor.is_user_idle():
                                for _file in filename:
                                    shot = [i for i in nuke.getFileNameList(os.path.dirname(self.path)) if _file in i]
                                    if shot:
                                        self.setup_read(os.path.join(os.path.dirname(self.path),shot[0]).replace('\\','/'), self.node)
                                count = 40
                        else:
                            if self.user_monitor.is_user_idle():
                                for _file in filename:
                                    with open(os.path.join(os.path.dirname(self.path), _file), 'r') as f:
                                        nuke.tcl('in root {%s}' % f.read())
                                count = 40
                    count +=1

        return self.timer_completed


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
is_correct_extension = any(shot_name.split('.')[-1].lower() in i for i in ('.png','.jpg', '.jpeg', '.exr', '.mov'))
is_mov = 'mov' in os.path.splitext(shot_name)[-1].lower() 
path_to_sequence_exists = os.path.isdir(node['path_to_sequence'].value())
is_node_connected = len(node.dependencies()) > 0
is_segmentation = node['model_to_run'].value() in ('sam','dam', 'sam3')
is_tracking = node['model_to_run'].value() == 'cotracker'
use_sam3 = node['use_sam3'].value() if node['model_to_run'].value() == 'sam3' else False
FG_render = node['fg_render'].value()


if is_segmentation or is_tracking:
    is_frame_legal = frame_idx >= int(node.firstFrame()) and frame_idx <= int(node.lastFrame())
    if node['use_limit'].value():
        is_frame_legal = frame_idx >= int(node['limit_first'].value()) and frame_idx <= int(node['limit_last'].value())
else:
    frame_idx = int(node.firstFrame()) if not node['use_limit'].value() else int(node['limit_first'].value()) 
    is_frame_legal = True
    
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
    config['id_class'] = node['id_class'].value() if node['id_class'].visible() else ''
    config['crop_position'] = [crop_position] if node['crop_position'].visible() else None
    config['path_to_sequence'] = node['path_to_sequence'].value()
    config['shot_name'] = shot_name
    config['render_to'] = node['render_to'].value()
    config['render_name'] = node['render_name'].value()
    config['use_gdino'] = node['use_gdino'].value()
    config['use_sam3'] = use_sam3
    config['is_mov'] = is_mov
    config['mov_last_frame'] = int(node.lastFrame() - node.firstFrame())
    config['frame_idx'] = int(frame_idx - node.firstFrame())
    config['first_frame_sequence'] = int(node.firstFrame())
    config['limit_range'] = node['use_limit'].value()
    config['delimiter'] = delimiter
    config['use_grid'] = node['use_grid'].value()
    config['grid_size'] = node['grid_size'].value()
    config['colourspace'] = 'linear' if node['model_to_run'].value() == 'rgb2x' else 'srgb'
    config['passes'] = ['albedo','roughness','metallic','normal','irradiance']
    if not node['all_passes'].value():
        pass_knobs = config['passes']
        config['passes'] = [i for i in pass_knobs if node[i].value() == True]


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
    keep_going = False
    if not node['skip_directory_checks'].value():
        _render_dir = config['render_to']
        if nuke.ask(f'Hey! Just checking that you are happy with the render location. This is where I am going to render your file. {_render_dir}'):
            keep_going =True 
    else:
        keep_going = True


    if keep_going:
        unique_name = str(uuid.uuid4())
        config['uuid'] = unique_name
        with open(os.path.join(config_save_directory, f'{unique_name}_{shot_name_no_padding}.json').replace('\\','/'), 'w') as f:
            json.dump(config, f, indent = 4)


        queue = node['queue'].value()
        queue = queue + ',' + unique_name if queue != '' else unique_name
        node['queue'].setValue(queue)
        render_progress_file = os.path.join(config['render_to'], f'{unique_name}_render_progress.json').replace('\\','/')
        if FG_render:
            nuke.thisNode().end()
            filename = run_progress_bar(render_progress_file, config['model_to_run'])
            if filename:
                if not is_tracking:
                    for read_idx, _file in enumerate(filename):
                        shot = [i for i in nuke.getFileNameList(os.path.dirname(render_progress_file)) if _file in i]
                        if shot:
                            read = nuke.nodes.Read()
                            read['file'].fromUserText(os.path.join(os.path.dirname(render_progress_file),shot[0]).replace('\\','/'))
                            read.setXYpos(node.xpos() + (100 * (read_idx + 1)) , node.ypos())
                else:
                    for tracker_idx, _file in enumerate(filename):
                        tracker = nuke.loadToolset(os.path.join(os.path.dirname(render_progress_file), _file))
                        tracker.setXYpos(node.xpos() + (100 * (tracker_idx + 1)) , node.ypos())

            remove_from_queue(node, render_progress_file)
            
        else:
            node_loader = Loader(node, render_progress_file, is_tracker = is_tracking)
else:
    error_message = '\n'.join(error_messages[error_keys[i]] for i in errors )
    nuke.alert(f'The following errors where found. Please ensure they are all fixed prior to writing the config:\n{error_message}')

