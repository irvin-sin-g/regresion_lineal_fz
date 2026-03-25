import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------------------
# CONTEXTO DE LA TAREA:
# Predecir el consumo de combustible (en litros) de una motocicleta
# YAMAHA FZ 2.0, basándose únicamente en la distancia recorrida (en km)
# a una velocidad constante.
# 
# ESPECIFICACIONES REALISTAS PARA FZ 2.0:
# - Rendimiento en condiciones óptimas: 45-50 km/litro
# - Consumo aproximado: 0.020 - 0.022 litros/km
# - Motor: 149cc, SOHC, refrigerado por aire
# --------------------------------------------------------------

# PASO 1: CREAR DATOS (Simulación del comportamiento de la FZ 2.0)
# Imagina que tenemos datos de 50 viajes diferentes con la moto.
# Característica (X): Distancia recorrida en kilómetros (entre 10 y 200 km)
# Objetivo (y): Combustible consumido en litros.
#   
# Para la FZ 2.0, el rendimiento óptimo es ~47.5 km/litro
# Esto significa: Consumo (L) = Distancia (km) / 47.5
# En términos lineales: Consumo = 0.02105 * Distancia
#
# Añadimos un pequeño ruido para simular:
# - Variaciones en la velocidad (no es perfectamente constante)
# - Condiciones del viento
# - Calidad del combustible
# - Ligera pendiente en la ruta

np.random.seed(42)  # Para resultados reproducibles

# Generamos 50 distancias de viaje aleatorias (en km)
X = np.random.randint(10, 200, size=(50, 1))

# Rendimiento teórico de la FZ 2.0 en condiciones ideales: 47.5 km/l
rendimiento_teorico_km_por_litro = 47.5
consumo_teorico_por_km = 1 / rendimiento_teorico_km_por_litro  # ≈ 0.02105 L/km

# Generamos el consumo base + ruido realista
# El ruido es proporcional a la distancia (mayor distancia = más variabilidad posible)
ruido = np.random.randn(50) * 0.3  # Desviación estándar de 0.3 litros
y = (consumo_teorico_por_km * X).squeeze() + ruido

print("=" * 60)
print("   EJERCICIO: Predicción de Consumo - YAMAHA FZ 2.0")
print("=" * 60)
print("\n📊 PRIMERAS 5 MUESTRAS DE DATOS DE VIAJES:")
print(f"   Distancia recorrida (km): {X[:5].flatten()}")
print(f"    Combustible consumido (L): {np.round(y[:5], 2)}")
print(f"\n   Rendimiento teórico base: {rendimiento_teorico_km_por_litro} km/litro")
print(f"   Consumo teórico base: {consumo_teorico_por_km:.4f} litros/km")


# PASO 2: DIVIDIR DATOS (Entrenamiento vs Prueba)
# Separamos los datos para evaluar el modelo con viajes que no ha "visto"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 DIVISIÓN DE DATOS:")
print(f"   Viajes para entrenar (Train): {len(X_train)} viajes")
print(f"   Viajes para validar (Test): {len(X_test)} viajes")


# PASO 3: ENTRENAR EL MODELO
# Usamos Regresión Lineal para encontrar la relación óptima
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Calculamos el rendimiento real del modelo (km/l)
consumo_aprendido_por_km = modelo.coef_[0]
rendimiento_aprendido_km_por_litro = 1 / consumo_aprendido_por_km if consumo_aprendido_por_km > 0 else 0

print(f"\n🔧 MODELO ENTRENADO CON ÉXITO:")
print(f"   Consumo por kilómetro aprendido: {consumo_aprendido_por_km:.4f} litros/km")
print(f"   📈 RENDIMIENTO ESTIMADO: {rendimiento_aprendido_km_por_litro:.1f} km/litro")
print(f"   Precisión del modelo: {modelo.score(X_train, y_train)*100:.1f}% en datos de entrenamiento")


# PASO 4: HACER PREDICCIONES Y EVALUAR
# Predecimos el consumo para los viajes de prueba
y_pred = modelo.predict(X_test)

print("\n🎯 COMPARACIÓN (Realidad vs Predicción) - VIAJES DE PRUEBA:")
print("-" * 50)
for i in range(5):
    distancia = X_test[i][0]
    real = y_test[i]
    predicho = y_pred[i]
    error = abs(real - predicho)
    print(f"Viaje {i+1}: {distancia:3.0f} km | Real: {real:5.2f} L | Predicho: {predicho:5.2f} L | Error: {error:.2f} L")

# Métricas de evaluación
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Raíz del error cuadrático medio (más interpretable)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 MÉTRICAS DE RENDIMIENTO:")
print(f"   Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"   Raíz del Error Cuadrático Medio (RMSE): {rmse:.3f} litros")
print(f"   Coeficiente de Determinación (R²): {r2:.4f}")
print(f"   Interpretación: El modelo explica el {r2*100:.1f}% de la variabilidad en el consumo")


# PASO 5: VISUALIZAR RESULTADOS
# Creamos una visualización profesional de los resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Datos reales vs línea de regresión
ax1.scatter(X_train, y_train, color='gray', alpha=0.5, label='Datos Entrenamiento', s=60)
ax1.scatter(X_test, y_test, color='blue', edgecolors='black', label='Datos Prueba (Real)', s=80, alpha=0.7)

# Línea de regresión
X_linea = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_linea = modelo.predict(X_linea)
ax1.plot(X_linea, y_linea, color='red', linewidth=2.5, label=f'Línea de Predicción\n({rendimiento_aprendido_km_por_litro:.1f} km/l)')

ax1.set_title('YAMAHA FZ 2.0: Predicción de Consumo de Combustible', fontsize=12, fontweight='bold')
ax1.set_xlabel('Distancia Recorrida (km)', fontsize=10)
ax1.set_ylabel('Combustible Consumido (litros)', fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 210])
ax1.set_ylim([0, 5.5])

# Gráfico 2: Predicciones vs Valores Reales (diagnóstico)
ax2.scatter(y_test, y_pred, color='green', alpha=0.6, s=80, edgecolors='black')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Predicción Perfecta')
ax2.set_title('Diagnóstico: Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
ax2.set_xlabel('Valores Reales (litros)', fontsize=10)
ax2.set_ylabel('Valores Predichos (litros)', fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# PASO 6: PRUEBAS CON ESCENARIOS REALES
print("\n" + "=" * 60)
print("   🏍️  PRUEBAS CON ESCENARIOS REALES PARA YAMAHA FZ 2.0")
print("=" * 60)

# Escenario 1: Viaje corto (25 km - viaje diario al trabajo)
distancia_corta = np.array([[25]])
consumo_corto = modelo.predict(distancia_corta)
print(f"\n📌 ESCENARIO 1: Viaje corto (25 km)")
print(f"   → Consumo estimado: {consumo_corto[0]:.2f} litros")
print(f"   → Rendimiento: {25/consumo_corto[0]:.1f} km/l")
print(f"   → Costo aproximado: ${consumo_corto[0]*22:.2f} MXN (gasolina ~$22/L)")

# Escenario 2: Viaje de fin de semana (120 km)
distancia_media = np.array([[120]])
consumo_medio = modelo.predict(distancia_media)
print(f"\n📌 ESCENARIO 2: Ruta de fin de semana (120 km)")
print(f"   → Consumo estimado: {consumo_medio[0]:.2f} litros")
print(f"   → Rendimiento: {120/consumo_medio[0]:.1f} km/l")
print(f"   → Costo aproximado: ${consumo_medio[0]*22:.2f} MXN")

# Escenario 3: Viaje largo (180 km - tanque lleno)
distancia_larga = np.array([[180]])
consumo_largo = modelo.predict(distancia_larga)
print(f"\n📌 ESCENARIO 3: Viaje largo (180 km)")
print(f"   → Consumo estimado: {consumo_largo[0]:.2f} litros")
print(f"   → Rendimiento: {180/consumo_largo[0]:.1f} km/l")
print(f"   → Comparación con tanque (12L): {12 - consumo_largo[0]:.1f} litros sobrantes")

# Consejo práctico
print(f"\n💡 CONSEJO PRÁCTICO:")
print(f"   Con la Yamaha FZ 2.0, puedes recorrer aproximadamente")
print(f"   {rendimiento_aprendido_km_por_litro * 12:.0f} km con un tanque lleno (12 litros).")
print(f"   ¡Ideal para rutas largas sin preocuparte por la gasolina!")