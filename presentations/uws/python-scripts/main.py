from manim import *
import numpy as np
class CreateSimpleExample(Scene):
    def construct(self):
        colorList = [RED, GREEN, BLUE, YELLOW]

        # Create the grid
        grid = Axes(
            x_range=[0, 10, .5],
            y_range=[0, 10, .5],
            x_length=9,
            y_length=5.5,
            axis_config={
                "numbers_to_include": np.arange(0, 11, 1),
                "font_size": 24,
            },
            tips=False,
        )

        y_label = grid.get_y_axis_label("y")
        x_label = grid.get_x_axis_label("x")

        self.play(Create(grid), Create(y_label), Create(x_label))

        points = []  # To store points and their target locations
        animations = []  # To store the animations for moving points

        for i in range(200):
            target_blue = [(np.random.random() * 7.8) - 3.9, (0.5 * np.random.random() * 3.5) + .1, 0]
            target_red = [(np.random.random() * 7.8) - 3.9, (0.5 * np.random.random() * -3.8) - .1, 0]

            blue_point = Dot(point=[0, 0, 0], color=BLUE, radius=.05)
            red_point = Dot(point=[0, 0, 0], color=RED, radius=.05)

            self.add(blue_point, red_point)

            # Store the points and their animations to move to target positions
            animations.append(blue_point.animate.move_to(target_blue))
            animations.append(red_point.animate.move_to(target_red))

        # Play animations to move points to their target positions
        self.play(*animations)
        self.wait(duration=1)


        text = Text("Impurity = 0.5").move_to([0,3,0])


        self.play(Create(text))

        self.wait(duration=1)

        self.wait(1)

        line = Line(start=[-4, 0,0], end=[4,0,0])
        self.play(Create(line))

        self.wait(2)


        t1 = Rectangle(width=8, height=0).set_fill(BLUE, opacity=.2)
        t2 = Rectangle(width=8).set_fill(BLUE, opacity=.3).move_to([0,1,0])

        m1 = Rectangle(width=8, height=0).set_fill(RED, opacity=.2)
        m2 = Rectangle(width=8).set_fill(RED, opacity=.3).move_to([0,-1,0])
        self.play(Transform(m1,m2), Transform(t1,t2))

        self.wait(duration=1)

        text3 = Text("Impurity = 0.0").move_to([0,3,0])
        self.play(Transform(text,text3))

        self.wait(duration=2)



class CreateMoreComplex(Scene):
    def construct(self):
        colorList = [RED, GREEN, BLUE, YELLOW]

        # Create the grid
        grid = Axes(
            x_range=[0, 10, .5],
            y_range=[0, 10, .5],
            x_length=9,
            y_length=5.5,
            axis_config={
                "numbers_to_include": np.arange(0, 11, 1),
                "font_size": 24,
            },
            tips=False,
        )

        y_label = grid.get_y_axis_label("y")
        x_label = grid.get_x_axis_label("x")

        self.play(Create(grid), Create(y_label), Create(x_label))

        points = []  # To store points and their target locations
        animations = []  # To store the animations for moving points

        for i in range(200):
            target_blue = [(np.random.random() * 7.8) - 3.9, (0.5 * np.random.random() * 3.5) + .1, 0]
            target_red = [(np.random.random() * (7.4 / 2)) - 3.9, (0.5 * np.random.random() * -3.5) - .1, 0]
            target_yellow = [(np.random.random() * (7.4 / 2)), (0.5 * np.random.random() * -3.5) - .1, 0]

            blue_point = Dot(point=[0, 0, 0], color=BLUE, radius=.05)
            red_point = Dot(point=[0, 0, 0], color=RED, radius=.05)
            yellow_point= Dot(point=[0, 0, 0], color=YELLOW, radius=.05)

            self.add(blue_point, red_point, yellow_point)

            # Store the points and their animations to move to target positions
            animations.append(blue_point.animate.move_to(target_blue))
            animations.append(red_point.animate.move_to(target_red))
            animations.append(yellow_point.animate.move_to(target_yellow))

        # Play animations to move points to their target positions
        self.play(*animations)
        self.wait(duration=1)


        text = Text("Impurity = 0.666").move_to([0,3,0])

        self.play(Create(text))

        self.wait(duration=1)

        self.wait(1)

        line = Line(start=[-4, 0,0], end=[4,0,0])
        self.play(Create(line))

        self.wait(2)


        t1 = Rectangle(width=8).set_fill(BLUE, opacity=.2).move_to([0,1,0])

        m1 = Rectangle(width=8).set_fill(RED, opacity=.2).move_to([0,-1,0])

        self.play(GrowFromCenter(t1))
        self.play(GrowFromCenter(m1))

        self.wait(duration=1)

        text3 = Text("Impurity = 0.5").move_to([0,3,0])
        self.play(Transform(text,text3))


        line = Line(start=[0, 0,0], end=[0,-2,0])
        self.play(Create(line))

        self.play(m1.animate.stretch(.5, dim=0).align_to([0, -1, 0], RIGHT))

        rb = Rectangle(width=4).set_fill(YELLOW, opacity=.3).align_to([0,-1,0], LEFT)
        rb.shift(DOWN)

        self.play(GrowFromCenter(rb))


        text4 = Text("Impurity = 0.0").move_to([0,3,0])
        self.play(Transform(text,text4))
        self.wait(duration=2)
