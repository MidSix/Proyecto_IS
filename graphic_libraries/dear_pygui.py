import dearpygui.dearpygui as dpg

# function to answer the user
def greet():
    name = dpg.get_value("name_input").strip()
    if name:
        dpg.set_value("greeting", f"Hello, {name}!")
    else:
        dpg.set_value("greeting","Please, write your name.")

# create context
dpg.create_context()

# window with fixed size
with dpg.window(label="Greeting App", width=300, height=200, no_resize=True, no_collapse=True, no_title_bar=True, no_move=True, no_scrollbar=True):
    # interface elements
    dpg.add_text("What's your name?")
    dpg.add_input_text(tag="name_input")
    dpg.add_button(label="Accept", callback=greet)
    dpg.add_text("", tag="greeting")

dpg.create_viewport(title='Greeting', width=300, height=200)
dpg.setup_dearpygui()
dpg.show_viewport() # show the window
dpg.start_dearpygui() # run the loop
dpg.destroy_context() # end context
