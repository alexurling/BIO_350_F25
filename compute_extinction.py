import numpy as np

# States: 0=Both occupied (B), 1=Small only (S), 2=Large only (L), 3=None (extinct, absorbing)
# Given per-year probabilities:
# local extinction: small patch e_s = 0.13, large patch e_l = 0.03
# recolonization when the other patch occupied: r = 0.02

e_s = 0.13
e_l = 0.03
r = 0.02

# Construct transition matrix P where rows sum to 1 and P[i,j] = P(next state = j | current state = i)
P = np.zeros((4,4))

# From Both occupied (B): possible events are extinction in small, large, both, or neither.
# We assume extinctions in two patches are independent within a year.
p_b_to_none = e_s * e_l  # both go extinct in same year
p_b_to_small = (1 - e_s) * e_l  # only large goes extinct -> small only occupied
p_b_to_large = e_s * (1 - e_l)  # only small goes extinct -> large only occupied
p_b_to_both = (1 - e_s) * (1 - e_l)  # neither goes extinct
P[0, 0] = p_b_to_both
P[0, 1] = p_b_to_small
P[0, 2] = p_b_to_large
P[0, 3] = p_b_to_none

# From Small-only (S): small occupied, large empty. Two events: small may go extinct; large may be recolonized from small.
# We'll assume recolonization happens from the occupied patch at rate r, independently of extinction.
# Outcomes:
# - If recolonization occurs (prob r): then next state is Both, regardless of whether small also goes extinct in the same year.
#   But if recolonization and small extinction both happen, we get Large-only. So need to enumerate combinations.
# Compute by cases:
# recolonize (R) or not (1-R), small goes extinct (E_s) or not (1-E_s).
P[1, 0] = r * (1 - e_s)        # R and small survives -> Both
P[1, 2] = r * e_s              # R and small extinct -> Large-only
P[1, 1] = (1 - r) * (1 - e_s)  # no R and small survives -> Small-only
P[1, 3] = (1 - r) * e_s        # no R and small extinct -> None

# From Large-only (L): symmetric to Small-only but swapping e_l and e_s
P[2, 0] = r * (1 - e_l)        # recolonize small and large survives -> Both
P[2, 1] = r * e_l              # recolonize and large goes extinct -> Small-only
P[2, 2] = (1 - r) * (1 - e_l)  # no recolonize and large survives -> Large-only
P[2, 3] = (1 - r) * e_l        # no recolonize and large extinct -> None

# From None (absorbing)
P[3, 3] = 1.0

# Verify rows sum to 1
row_sums = P.sum(axis=1)
assert np.allclose(row_sums, 1.0), f"Rows sums not 1: {row_sums}"

# Compute distribution after t years, starting from Both occupied
initial = np.array([1.0, 0.0, 0.0, 0.0])
T = 50
state = initial.copy()
for year in range(T):
    state = state.dot(P)

prob_extinct_by_50 = state[3]

print(f"Probability species permanently lost by year {T}: {prob_extinct_by_50:.6f} ({prob_extinct_by_50*100:.4f}%)")
