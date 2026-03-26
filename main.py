<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimplexBigM:
    """
    Class to solve Linear Programming problems using Big M Simplex method.
    """
    def __init__(self, num_vars, constraints, obj_type, obj_coeffs):
        self.num_vars = num_vars
        self.constraints = constraints
        self.obj_type = obj_type
        self.obj_coeffs = obj_coeffs
        self.M = 1000000  # Large penalty for artificial variables
        self.tableau = None
        self.basis = []
        self.all_vars = []
        self.solution = None
        self.obj_value = None
        self.setup_tableau()

    def setup_tableau(self):
        """
        Set up the initial simplex tableau with Big M method.
        """
        # Count additional variables
        slack_count = sum(1 for _, t, _ in self.constraints if t == '<=')
        surplus_count = sum(1 for _, t, _ in self.constraints if t == '>=')
        artificial_count = sum(1 for _, t, _ in self.constraints if t in ['>=', '='])

        self.all_vars = ([f'x{i+1}' for i in range(self.num_vars)] +
                         [f's{i+1}' for i in range(slack_count)] +
                         [f'su{i+1}' for i in range(surplus_count)] +
                         [f'a{i+1}' for i in range(artificial_count)])

        num_rows = len(self.constraints) + 1  # +1 for objective row
        num_cols = len(self.all_vars) + 1  # +1 for RHS
        self.tableau = np.zeros((num_rows, num_cols))

        # Fill constraint rows
        var_idx = self.num_vars
        for i, (coeffs, type_, rhs) in enumerate(self.constraints):
            for j in range(self.num_vars):
                self.tableau[i, j] = coeffs[j]
            self.tableau[i, -1] = rhs
            if type_ == '<=':
                self.tableau[i, var_idx] = 1
                self.basis.append(var_idx)
                var_idx += 1
            elif type_ == '>=':
                self.tableau[i, var_idx] = -1  # surplus
                var_idx += 1
                self.tableau[i, var_idx] = 1  # artificial
                self.basis.append(var_idx)
                var_idx += 1
            elif type_ == '=':
                self.tableau[i, var_idx] = 1  # artificial
                self.basis.append(var_idx)
                var_idx += 1

        # Objective row coefficients (minimization orientation in tableau)
        obj_row = len(self.constraints)
        if self.obj_type == 'max':
            obj_coeffs = [-c for c in self.obj_coeffs]
        else:
            obj_coeffs = [c for c in self.obj_coeffs]

        for j in range(self.num_vars):
            self.tableau[obj_row, j] = obj_coeffs[j]

        # Add Big M penalty for artificial variables
        for idx, var in enumerate(self.all_vars):
            if var.startswith('a'):
                self.tableau[obj_row, idx] = self.M

        # If artificial vars are in basis, eliminate them from objective row using Big M method
        for i, basic_idx in enumerate(self.basis):
            if self.all_vars[basic_idx].startswith('a'):
                self.tableau[obj_row] -= self.M * self.tableau[i]

    def get_tableau_df(self):
        """
        Get the current tableau as a pandas DataFrame.
        """
        columns = self.all_vars + ['RHS']
        df = pd.DataFrame(self.tableau, columns=columns)
        return df

    def simplex(self, tableaux=None):
        """
        Perform simplex iterations until optimal solution or detect unbounded/infeasible.
        If tableaux is provided, append DataFrames to it.
        Returns 'unbounded', 'infeasible', or (solution, obj_value)
        """
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        while iteration < max_iterations:
            if tableaux is not None:
                tableaux.append(self.get_tableau_df())

            # Check optimality: all coefficients in objective row >= 0 (except artificials, but M is large)
            obj_coeffs = self.tableau[-1, :-1]
            if all(x >= -1e-6 for x in obj_coeffs):  # Allow small numerical errors
                break

            # Entering variable: smallest index with negative coefficient (Bland's rule)
            entering = None
            for j in range(len(obj_coeffs)):
                if obj_coeffs[j] < -1e-6:
                    entering = j
                    break
            if entering is None:
                break  # Optimal

            # Check for unbounded: no positive coefficients in entering column
            if all(self.tableau[i, entering] <= 1e-6 for i in range(len(self.constraints))):
                return 'unbounded'

            # Leaving variable: minimum ratio test, smallest index (Bland's rule)
            ratios = []
            for i in range(len(self.constraints)):
                if self.tableau[i, entering] > 1e-6:
                    ratio = self.tableau[i, -1] / self.tableau[i, entering]
                    ratios.append((ratio, i))
                else:
                    ratios.append((float('inf'), i))
            if all(r[0] == float('inf') for r in ratios):
                return 'unbounded'
            min_ratio = min(r[0] for r in ratios if r[0] != float('inf'))
            leaving = min(i for ratio, i in ratios if ratio == min_ratio)

            # Pivot
            pivot_row = leaving
            pivot_col = entering
            pivot_val = self.tableau[pivot_row, pivot_col]

            # Normalize pivot row
            self.tableau[pivot_row] /= pivot_val

            # Eliminate other rows
            for i in range(len(self.tableau)):
                if i != pivot_row:
                    factor = self.tableau[i, pivot_col]
                    self.tableau[i] -= factor * self.tableau[pivot_row]

            # Update basis
            self.basis[pivot_row] = pivot_col
            iteration += 1

        if iteration >= max_iterations:
            return 'max_iterations_exceeded'

        # Check for infeasibility: artificial variables in basis with positive values
        for i, var_idx in enumerate(self.basis):
            if self.all_vars[var_idx].startswith('a') and self.tableau[i, -1] > 1e-6:
                return 'infeasible'

        # Extract solution
        self.solution = {f'x{i+1}': 0.0 for i in range(self.num_vars)}
        for i, var_idx in enumerate(self.basis):
            var_name = self.all_vars[var_idx]
            if var_name.startswith('x'):
                self.solution[var_name] = self.tableau[i, -1]

        self.obj_value = self.tableau[-1, -1]
        return self.solution, self.obj_value

    def visualize(self):
        """
        Visualize the objective function in 3D for 2-variable problems.
        Returns the figure.
        """
        if self.num_vars != 2 or self.solution is None:
            return None

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create grid
        x1_range = np.linspace(0, max(10, self.solution['x1'] * 1.5), 50)
        x2_range = np.linspace(0, max(10, self.solution['x2'] * 1.5), 50)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        # Objective function
        Z = self.obj_coeffs[0] * X1 + self.obj_coeffs[1] * X2

        # Plot surface
        ax.plot_surface(X1, X2, Z, alpha=0.7, cmap='viridis')

        # Plot optimal point
        ax.scatter(self.solution['x1'], self.solution['x2'], self.obj_value,
                  color='red', s=100, label='Optimal Point')

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Objective Value')
        ax.set_title('3D Visualization of Objective Function')
        ax.legend()
        return fig

def main():
    """
    Main Streamlit app function.
    """
    st.title("Linear Programming Solver with Big M Method")

    with st.form("lp_form"):
        st.header("Problem Setup")
        num_vars = st.number_input("Number of variables", min_value=1, value=2, step=1)
        num_constraints = st.number_input("Number of constraints", min_value=1, value=2, step=1)
        obj_type = st.selectbox("Objective type", ["max", "min"])
        obj_coeffs_str = st.text_input("Objective coefficients (space separated)", "3 2")

        st.header("Constraints")
        constraints_input = []
        for i in range(int(num_constraints)):
            st.subheader(f"Constraint {i+1}")
            coeffs_str = st.text_input(f"Coefficients for constraint {i+1} (space separated)", "1 1", key=f"coeffs_{i}")
            type_ = st.selectbox(f"Type for constraint {i+1}", ["<=", ">=", "="], key=f"type_{i}")
            rhs = st.number_input(f"RHS for constraint {i+1}", value=4.0, key=f"rhs_{i}")
            constraints_input.append((coeffs_str, type_, rhs))

        submitted = st.form_submit_button("Solve")

    if submitted:
        try:
            # Parse inputs
            obj_coeffs = list(map(float, obj_coeffs_str.split()))
            if len(obj_coeffs) != int(num_vars):
                st.error("Number of objective coefficients must match number of variables")
                return

            constraints = []
            for coeffs_str, type_, rhs in constraints_input:
                coeffs = list(map(float, coeffs_str.split()))
                if len(coeffs) != int(num_vars):
                    st.error(f"Number of coefficients in constraint must match number of variables")
                    return
                constraints.append((coeffs, type_, rhs))

            # Solve
            solver = SimplexBigM(int(num_vars), constraints, obj_type, obj_coeffs)
            tableaux = []
            result = solver.simplex(tableaux)

            # Display tableaux
            st.header("Simplex Tableaux")
            for i, df in enumerate(tableaux):
                st.subheader(f"Iteration {i}")
                st.dataframe(df.style.format("{:.4f}"))

            if isinstance(result, str):
                st.error(f"No optimal solution: {result.replace('_', ' ').title()}")
            else:
                sol, val = result
                st.success("Optimal Solution Found")
                st.subheader("Solution")
                for var, v in sol.items():
                    st.write(f"{var} = {v:.4f}")
                st.write(f"**Objective Value:** {val:.4f}")

                # Visualize if 2 vars
                if int(num_vars) == 2:
                    fig = solver.visualize()
                    if fig:
                        st.subheader("3D Visualization")
                        st.pyplot(fig)

        except ValueError as e:
            st.error(f"Input error: {e}")

if __name__ == "__main__":
=======
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimplexBigM:
    """
    Class to solve Linear Programming problems using Big M Simplex method.
    """
    def __init__(self, num_vars, constraints, obj_type, obj_coeffs):
        self.num_vars = num_vars
        self.constraints = constraints
        self.obj_type = obj_type
        self.obj_coeffs = obj_coeffs
        self.M = 1000000  # Large penalty for artificial variables
        self.tableau = None
        self.basis = []
        self.all_vars = []
        self.solution = None
        self.obj_value = None
        self.setup_tableau()

    def setup_tableau(self):
        """
        Set up the initial simplex tableau with Big M method.
        """
        # Count additional variables
        slack_count = sum(1 for _, t, _ in self.constraints if t == '<=')
        surplus_count = sum(1 for _, t, _ in self.constraints if t == '>=')
        artificial_count = sum(1 for _, t, _ in self.constraints if t in ['>=', '='])

        self.all_vars = ([f'x{i+1}' for i in range(self.num_vars)] +
                         [f's{i+1}' for i in range(slack_count)] +
                         [f'su{i+1}' for i in range(surplus_count)] +
                         [f'a{i+1}' for i in range(artificial_count)])

        num_rows = len(self.constraints) + 1  # +1 for objective row
        num_cols = len(self.all_vars) + 1  # +1 for RHS
        self.tableau = np.zeros((num_rows, num_cols))

        # Fill constraint rows
        var_idx = self.num_vars
        for i, (coeffs, type_, rhs) in enumerate(self.constraints):
            for j in range(self.num_vars):
                self.tableau[i, j] = coeffs[j]
            self.tableau[i, -1] = rhs
            if type_ == '<=':
                self.tableau[i, var_idx] = 1
                self.basis.append(var_idx)
                var_idx += 1
            elif type_ == '>=':
                self.tableau[i, var_idx] = -1  # surplus
                var_idx += 1
                self.tableau[i, var_idx] = 1  # artificial
                self.basis.append(var_idx)
                var_idx += 1
            elif type_ == '=':
                self.tableau[i, var_idx] = 1  # artificial
                self.basis.append(var_idx)
                var_idx += 1

        # Objective row coefficients (minimization orientation in tableau)
        obj_row = len(self.constraints)
        if self.obj_type == 'max':
            obj_coeffs = [-c for c in self.obj_coeffs]
        else:
            obj_coeffs = [c for c in self.obj_coeffs]

        for j in range(self.num_vars):
            self.tableau[obj_row, j] = obj_coeffs[j]

        # Add Big M penalty for artificial variables
        for idx, var in enumerate(self.all_vars):
            if var.startswith('a'):
                self.tableau[obj_row, idx] = self.M

        # If artificial vars are in basis, eliminate them from objective row using Big M method
        for i, basic_idx in enumerate(self.basis):
            if self.all_vars[basic_idx].startswith('a'):
                self.tableau[obj_row] -= self.M * self.tableau[i]

    def get_tableau_df(self):
        """
        Get the current tableau as a pandas DataFrame.
        """
        columns = self.all_vars + ['RHS']
        df = pd.DataFrame(self.tableau, columns=columns)
        return df

    def simplex(self, tableaux=None):
        """
        Perform simplex iterations until optimal solution or detect unbounded/infeasible.
        If tableaux is provided, append DataFrames to it.
        Returns 'unbounded', 'infeasible', or (solution, obj_value)
        """
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        while iteration < max_iterations:
            if tableaux is not None:
                tableaux.append(self.get_tableau_df())

            # Check optimality: all coefficients in objective row >= 0 (except artificials, but M is large)
            obj_coeffs = self.tableau[-1, :-1]
            if all(x >= -1e-6 for x in obj_coeffs):  # Allow small numerical errors
                break

            # Entering variable: smallest index with negative coefficient (Bland's rule)
            entering = None
            for j in range(len(obj_coeffs)):
                if obj_coeffs[j] < -1e-6:
                    entering = j
                    break
            if entering is None:
                break  # Optimal

            # Check for unbounded: no positive coefficients in entering column
            if all(self.tableau[i, entering] <= 1e-6 for i in range(len(self.constraints))):
                return 'unbounded'

            # Leaving variable: minimum ratio test, smallest index (Bland's rule)
            ratios = []
            for i in range(len(self.constraints)):
                if self.tableau[i, entering] > 1e-6:
                    ratio = self.tableau[i, -1] / self.tableau[i, entering]
                    ratios.append((ratio, i))
                else:
                    ratios.append((float('inf'), i))
            if all(r[0] == float('inf') for r in ratios):
                return 'unbounded'
            min_ratio = min(r[0] for r in ratios if r[0] != float('inf'))
            leaving = min(i for ratio, i in ratios if ratio == min_ratio)

            # Pivot
            pivot_row = leaving
            pivot_col = entering
            pivot_val = self.tableau[pivot_row, pivot_col]

            # Normalize pivot row
            self.tableau[pivot_row] /= pivot_val

            # Eliminate other rows
            for i in range(len(self.tableau)):
                if i != pivot_row:
                    factor = self.tableau[i, pivot_col]
                    self.tableau[i] -= factor * self.tableau[pivot_row]

            # Update basis
            self.basis[pivot_row] = pivot_col
            iteration += 1

        if iteration >= max_iterations:
            return 'max_iterations_exceeded'

        # Check for infeasibility: artificial variables in basis with positive values
        for i, var_idx in enumerate(self.basis):
            if self.all_vars[var_idx].startswith('a') and self.tableau[i, -1] > 1e-6:
                return 'infeasible'

        # Extract solution
        self.solution = {f'x{i+1}': 0.0 for i in range(self.num_vars)}
        for i, var_idx in enumerate(self.basis):
            var_name = self.all_vars[var_idx]
            if var_name.startswith('x'):
                self.solution[var_name] = self.tableau[i, -1]

        self.obj_value = self.tableau[-1, -1]
        return self.solution, self.obj_value

    def visualize(self):
        """
        Visualize the objective function in 3D for 2-variable problems.
        Returns the figure.
        """
        if self.num_vars != 2 or self.solution is None:
            return None

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create grid
        x1_range = np.linspace(0, max(10, self.solution['x1'] * 1.5), 50)
        x2_range = np.linspace(0, max(10, self.solution['x2'] * 1.5), 50)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        # Objective function
        Z = self.obj_coeffs[0] * X1 + self.obj_coeffs[1] * X2

        # Plot surface
        ax.plot_surface(X1, X2, Z, alpha=0.7, cmap='viridis')

        # Plot optimal point
        ax.scatter(self.solution['x1'], self.solution['x2'], self.obj_value,
                  color='red', s=100, label='Optimal Point')

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Objective Value')
        ax.set_title('3D Visualization of Objective Function')
        ax.legend()
        return fig

def main():
    """
    Main Streamlit app function.
    """
    st.title("Linear Programming Solver with Big M Method")

    with st.form("lp_form"):
        st.header("Problem Setup")
        num_vars = st.number_input("Number of variables", min_value=1, value=2, step=1)
        num_constraints = st.number_input("Number of constraints", min_value=1, value=2, step=1)
        obj_type = st.selectbox("Objective type", ["max", "min"])
        obj_coeffs_str = st.text_input("Objective coefficients (space separated)", "3 2")

        st.header("Constraints")
        constraints_input = []
        for i in range(int(num_constraints)):
            st.subheader(f"Constraint {i+1}")
            coeffs_str = st.text_input(f"Coefficients for constraint {i+1} (space separated)", "1 1", key=f"coeffs_{i}")
            type_ = st.selectbox(f"Type for constraint {i+1}", ["<=", ">=", "="], key=f"type_{i}")
            rhs = st.number_input(f"RHS for constraint {i+1}", value=4.0, key=f"rhs_{i}")
            constraints_input.append((coeffs_str, type_, rhs))

        submitted = st.form_submit_button("Solve")

    if submitted:
        try:
            # Parse inputs
            obj_coeffs = list(map(float, obj_coeffs_str.split()))
            if len(obj_coeffs) != int(num_vars):
                st.error("Number of objective coefficients must match number of variables")
                return

            constraints = []
            for coeffs_str, type_, rhs in constraints_input:
                coeffs = list(map(float, coeffs_str.split()))
                if len(coeffs) != int(num_vars):
                    st.error(f"Number of coefficients in constraint must match number of variables")
                    return
                constraints.append((coeffs, type_, rhs))

            # Solve
            solver = SimplexBigM(int(num_vars), constraints, obj_type, obj_coeffs)
            tableaux = []
            result = solver.simplex(tableaux)

            # Display tableaux
            st.header("Simplex Tableaux")
            for i, df in enumerate(tableaux):
                st.subheader(f"Iteration {i}")
                st.dataframe(df.style.format("{:.4f}"))

            if isinstance(result, str):
                st.error(f"No optimal solution: {result.replace('_', ' ').title()}")
            else:
                sol, val = result
                st.success("Optimal Solution Found")
                st.subheader("Solution")
                for var, v in sol.items():
                    st.write(f"{var} = {v:.4f}")
                st.write(f"**Objective Value:** {val:.4f}")

                # Visualize if 2 vars
                if int(num_vars) == 2:
                    fig = solver.visualize()
                    if fig:
                        st.subheader("3D Visualization")
                        st.pyplot(fig)

        except ValueError as e:
            st.error(f"Input error: {e}")

if __name__ == "__main__":
>>>>>>> 69c6db26eb965c8504c6f05041f5b37e17c8a992
    main()