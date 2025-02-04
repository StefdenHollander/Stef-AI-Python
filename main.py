# Stap 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Stap 3
#data
studie_uren = np.array([1, 2, 3, 4, 5]). reshape(-1, 1)
oefentoetsen = np.array([2, 3, 3, 4, 5]). reshape(-1, 1)
motivation = np.array([0.2, 0.2, 0.2, 0.2, 0.2]). reshape(-1, 1)
cijfers = np.array([3, 4.5, 5, 6.5, 8])

# Stap 4
# Combineer de twee variabelen tot een matrix van inputvariabelen

X = np.hstack((studie_uren, oefentoetsen, motivation))

# Stap 5
#  Model aanmaken en trainen
model = LinearRegression()
model.fit(X, cijfers)

# Stap 6
#voorspellingen maken
predictions = model.predict(X)

# Stap 7
# Coefficienten en intercept
print("Intercept", model.intercept_)
print("Coefficients:", model.coef_) # b1 en b2

# Voorspellingen weergeven
print("Voorspelde cijfers", predictions)

# Stap 8
# Plotten en voorspellingen en werkelijke gegevens
fig, ax = plt.subplots()
ax.scatter(range(len(cijfers)), cijfers, color='blue', label='Werkelijke cijfers')
ax.plot(range(len(cijfers)), predictions, color='red', label='Voorspelde cijfers')
plt.xlabel('Studenten (volgorde)')
plt.ylabel('Cijfers')
plt.legend()
plt.show()