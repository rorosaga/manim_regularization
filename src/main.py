from manim import *

class RegularizationDemo(Scene):
    def construct(self):
        # Title
        title = Text("Regularization in Machine Learning", font_size=36).to_edge(UP)
        self.play(Write(title))

        # Define Axes
        axes = Axes(
            x_range=[-3, 3], y_range=[-3, 3], 
            axis_config={"include_tip": False}
        ).shift(DOWN)

        x_label = Text("x", font_size=24).next_to(axes.x_axis, DOWN)
        y_label = Text("y", font_size=24).next_to(axes.y_axis, LEFT)
        self.play(Create(axes), Write(x_label), Write(y_label))

        # Sample dataset (noisy points)
        data_points = [
            Dot(axes.c2p(x, 0.5 * x**3 - x + 0.3 * (-1) ** i), color=WHITE)
            for i, x in enumerate([-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5])
        ]
        self.play(*[FadeIn(dot) for dot in data_points])

        # Overfitted model: High-degree polynomial
        overfit_curve = axes.plot(lambda x: 0.5 * x**5 - 2 * x**3 + x, color=RED)
        overfit_label = Text("Overfitted Model", color=RED, font_size=24).next_to(overfit_curve, RIGHT)

        self.play(Create(overfit_curve), Write(overfit_label))
        self.wait(2)

        # Regularized model: Lower-degree polynomial (L2 Regularization)
        regularized_curve = axes.plot(lambda x: 0.2 * x**3 - 1.2 * x, color=BLUE)
        regularized_label = Text("L2 Regularization (Ridge)", color=BLUE, font_size=24).next_to(regularized_curve, LEFT)

        self.play(Transform(overfit_curve, regularized_curve), Transform(overfit_label, regularized_label))
        self.wait(2)

        # L1 Regularization: More Sparse Model
        l1_curve = axes.plot(lambda x: 0.8 * x, color=GREEN)
        l1_label = Text("L1 Regularization (Lasso)", color=GREEN, font_size=24).next_to(l1_curve, UP)

        self.play(Transform(overfit_curve, l1_curve), Transform(overfit_label, l1_label))
        self.wait(2)

        # Final message
        conclusion = Text("Regularization prevents overfitting and improves generalization!", font_size=30).to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(3)

        # Fade out all elements
        self.play(
            FadeOut(title, axes, x_label, y_label, *data_points, overfit_curve, overfit_label, conclusion)
        )


if __name__ == "__main__":
    scene = RegularizationDemo()
    scene.render()

