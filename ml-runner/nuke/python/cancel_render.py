import os
import nuke

node = nuke.thisNode()
config_save_directory = node["render_to"].value()

with open(os.path.join(config_save_directory, "cancel_render"), "w") as f:
    f.write("cancel")
