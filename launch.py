import evolution
import moderngl as mgl
import moderngl_window as mglw

import evolution.gworld
import evolution.visual.main
import torch
torch.set_grad_enabled(False)
# torch.random.manual_seed(0)

# Blocking call entering rendering/event loop
# mglw.run_window_config(evolution.visual.Game)

evolution.visual.main.main()

# evolution.gworld.main()