import evolution
import moderngl as mgl
import moderngl_window as mglw

import evolution.visual


# Blocking call entering rendering/event loop
mglw.run_window_config(evolution.visual.Game)