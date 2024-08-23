import glm
from sdl2 import timer
from typing import TYPE_CHECKING
import moderngl_window as mglw

from .camera import Camera

if TYPE_CHECKING:
    from evolution.visual.main import Game

class Controller:
    def __init__(self, window: mglw.BaseWindow, camera: Camera, game: 'Game'):
        # add all attrs of controller that end in _func to window, since these are associated with events
        for attr in dir(self):
            if attr.endswith('_func'):
                setattr(window, attr, getattr(self, attr))

        self.wnd = window
        self.now = timer.SDL_GetTicks()
        self.delta_time = 0
        self.camera = camera
        self.game = game
        self.click_pos = None
        self.mouse_pressed = False
        self.mouse_pos = None
        
    def key_event_func(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.W:   # movement
                self.camera.process_keyboard('FORWARD', self.delta_time)

            if key == self.wnd.keys.S:   # movement
                self.camera.process_keyboard('BACKWARD', self.delta_time)

            if key == self.wnd.keys.A:   # movement
                self.camera.process_keyboard('LEFT', self.delta_time)
                
            if key == self.wnd.keys.H:   # toggle creature appearances
                self.game.creatures.toggle_visibility()
                self.set_selected_creature(None)

            if key == self.wnd.keys.D:   # toggle extra food decay
                # self.camera.process_keyboard('RIGHT', self.delta_time)
                self.game.world.toggle_increasing_food_decr()
                print(self.game.cfg.food_cover_decr)

            if key == self.wnd.keys.R:   # reset camera to starting position, deselect any creature
                self.set_selected_creature(None)
                self.camera.reset_camera()

            if key == self.wnd.keys.Y:   # toggle whether we are locked on to the selected creature
                self.camera.toggle_follow()

            if key == self.wnd.keys.T:   # deselect creature
                self.set_selected_creature(None)

            if key == self.wnd.keys.SPACE:   # pause simulation
                self.game.toggle_pause()

            if key == self.wnd.keys.RIGHT:   # force step simulation by 1 
                self.game.step(n=1, force=True)

            if key == self.wnd.keys.C:    # save snapshot of current state to 'game.ckpt'
                self.game.save()

            if key == self.wnd.keys.NUMBER_0:   # start following creature 0 
                self.set_selected_creature(0)

            if key == self.wnd.keys.P:     # increase id of followed creature by 1
                self.set_selected_creature(self.game.world.creatures._selected_creature + 1)
                
            if key == self.wnd.keys.G:    # jump to Genghis Khan (guy with most offspring)
                most_kids = self.game.world.creatures.n_children.argmax()
                self.set_selected_creature(most_kids)
                
            if key == self.wnd.keys.O:   # jump to oldest creature
                oldest = self.game.world.creatures.ages.argmax()
                self.set_selected_creature(oldest)

            if key == self.wnd.keys.UP:    # speed up simulation so that we do 1 more step per frame
                amt = 1
                if modifiers.shift:
                    amt *= 10
                if modifiers.ctrl:
                    amt *= 10
                self.game.game_speed += amt

            if key == self.wnd.keys.DOWN:  # slow down simulation so that we do 1 less step per frame
                amt = 1
                if modifiers.shift:
                    amt *= 10
                if modifiers.ctrl:
                    amt *= 10
                self.game.game_speed -= amt
                self.game.game_speed = max(1, self.game.game_speed)
 
            if key == self.wnd.keys.B:    # turn hitboxes on/off
                self.game.creatures.toggle_hitboxes()

    def set_selected_creature(self, creature):
        print(f"Setting to creature {creature}"
              f"\n\tPos: {self.game.world.creatures.positions[creature]}"
              f"\n\tSize: {self.game.world.creatures.sizes[creature]}"
              f"\n\tHealth: {self.game.world.creatures.healths[creature]}"
              f"\n\tEnergy: {self.game.world.creatures.energies[creature]}")
        self.camera.following = True
        self.game.world.set_selected_creature(creature)
        # if creature is not None:
        #     print('collision', self.game.world.collisions[creature])
        #     print('rays', self.game.world.creatures.rays[creature])
        #     posn = self.game.world.creatures.positions[creature]
        #     # print('position', posn)
        #     grid_posn = (posn // self.game.world.cfg.cell_size).long()
        #     # print('cell_size', self.game.world.cfg.cell_size)
        #     # print('grid_center',self.game.world.celled_world[1].shape, self.game.world.celled_world[1][grid_posn[1], grid_posn[0]])
        #     # print('grid_position', self.game.world.celled_world[1][grid_posn[1]-4:grid_posn[1]+4, grid_posn[0]-4:grid_posn[0]+4])
        
    def set_selected_cell(self, xy):
        self.game.world.set_selected_cell(xy)
        
    def mouse_press_event_func(self, x, y, button):
        # need to store everything in pixel coords so that we are closest to the actual input 
        # this avoids jittering and weird camera behavior when dragging the map
        # print('posn', self.game.world.creatures.positions)
        # print('size', self.game.world.creatures.sizes)
        # print('color', self.game.world.creatures.colors)
        # print('head_dir', self.game.world.creatures.head_dirs)
        # print('rays', self.game.world.creatures.rays)
        if button == self.wnd.mouse.left:
            self.click_pos = glm.vec2(x, y)
            self.camera_pos = self.camera.position.xy
            self.mouse_pressed = True
            game_click = self.camera.pixel_to_game_coords(*self.click_pos).xy
            creature_id = self.game.world.click_creature(game_click) if self.game.creatures.visible else None
            if creature_id is not None:
                self.set_selected_creature(creature_id)
            elif self.camera.click_in_bounds(game_click):
                self.set_selected_cell(game_click)
            else:
                self.set_selected_cell(None)
        # print(self.camera.cfg.food_cover_decr)
        
        # print(self.game.world.creatures.positions)
        # print(self.game.world.creatures.sizes)
            # print(game_click, self.game.world.creatures.positions, self.game.world.creatures.rays, creature)
            # if creature is not None:
            # self.game.select_creature(creature)
            
    def mouse_scroll_event_func(self, xoffset, yoffset):
        self.camera.zoom_into_point(yoffset, self.mouse_pos)

    def mouse_release_event_func(self, x, y, button):
        self.mouse_pressed = False

    def mouse_drag_event_func(self, x, y, dx, dy):
        if not self.mouse_pressed:
            return
        self.camera.drag(self.camera_pos, self.click_pos, glm.vec2(x, y))

    def mouse_position_event_func(self, x, y, dx, dy):
        self.mouse_pos = glm.vec2(x, y)

    def tick(self):
        curr_time = timer.SDL_GetTicks()
        self.delta_time = (curr_time - self.now) / 1000.0
        self.now = curr_time