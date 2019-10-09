"""Binary tree with decision tree semantics and ASCII visualization."""


class Node:
    """A decision tree node."""

    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

    def debug(self, feature_names, class_names, show_details):
        """Print an ASCII visualization of the tree."""
        lines, _, _, _ = self._debug_aux(
            feature_names, class_names, show_details, root=True
        )
        for line in lines:
            print(line)

    def _debug_aux(self, feature_names, class_names, show_details, root=False):
        # See https://stackoverflow.com/a/54074933/1143396 for similar code.
        is_leaf = not self.right
        if is_leaf:
            lines = [class_names[self.predicted_class]]
        else:
            lines = [
                "{} < {:.2f}".format(feature_names[self.feature_index], self.threshold)
            ]
        if show_details:
            lines += [
                "gini = {:.2f}".format(self.gini),
                "samples = {}".format(self.num_samples),
                str(self.num_samples_per_class),
            ]
        width = max(len(line) for line in lines)
        height = len(lines)
        if is_leaf:
            lines = ["║ {:^{width}} ║".format(line, width=width) for line in lines]
            lines.insert(0, "╔" + "═" * (width + 2) + "╗")
            lines.append("╚" + "═" * (width + 2) + "╝")
        else:
            lines = ["│ {:^{width}} │".format(line, width=width) for line in lines]
            lines.insert(0, "┌" + "─" * (width + 2) + "┐")
            lines.append("└" + "─" * (width + 2) + "┘")
            lines[-2] = "┤" + lines[-2][1:-1] + "├"
        width += 4  # for padding

        if is_leaf:
            middle = width // 2
            lines[0] = lines[0][:middle] + "╧" + lines[0][middle + 1 :]
            return lines, width, height, middle

        # If not a leaf, must have two children.
        left, n, p, x = self.left._debug_aux(feature_names, class_names, show_details)
        right, m, q, y = self.right._debug_aux(feature_names, class_names, show_details)
        top_lines = [n * " " + line + m * " " for line in lines[:-2]]
        # fmt: off
        middle_line = x * " " + "┌" + (n - x - 1) * "─" + lines[-2] + y * "─" + "┐" + (m - y - 1) * " "
        bottom_line = x * " " + "│" + (n - x - 1) * " " + lines[-1] + y * " " + "│" + (m - y - 1) * " "
        # fmt: on
        if p < q:
            left += [n * " "] * (q - p)
        elif q < p:
            right += [m * " "] * (p - q)
        zipped_lines = zip(left, right)
        lines = (
            top_lines
            + [middle_line, bottom_line]
            + [a + width * " " + b for a, b in zipped_lines]
        )
        middle = n + width // 2
        if not root:
            lines[0] = lines[0][:middle] + "┴" + lines[0][middle + 1 :]
        return lines, n + m + width, max(p, q) + 2 + len(top_lines), middle
