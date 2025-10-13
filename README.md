linear

Sehr stark, Abi 🔥 — du hast hier im Prinzip schon ein Mini-Model-Benchmarking-System gebaut, und die Ergebnisse lassen sich sehr klar interpretieren, wenn man weiß, was die Kennzahlen bedeuten.

Lass uns das gemeinsam präzise und fachlich sauber durchgehen 👇

📊 Ergebnisse im Überblick
Modell	MAE	MSE	R²
Linear Regression	3.60	77.37	0.800
Ridge Regression	2.90	87.56	0.773
Lasso Regression	3.03	90.83	0.765
Elastic Net Regression	2.97	88.93	0.770
🧠 1️⃣ Verständnis der Metriken

MAE (Mean Absolute Error): Durchschnittlicher absoluter Fehler → kleiner = besser

MSE (Mean Squared Error): Bestraft große Fehler stärker → kleiner = besser

R² (Bestimmtheitsmaß): Anteil der erklärten Varianz → größer = besser (1.0 = perfekt)

🧩 2️⃣ Bewertung pro Modell
⚙️ Linear Regression

R² = 0.80 → Bestes Ergebnis

MAE = 3.6 → leicht höherer Fehler

Kein Regularisierungsterm → passt die Daten perfekt linear an.
→ Kann bei vielen korrelierten oder redundanten Features instabil werden.
→ Aber wenn deine Daten wirklich sauber und relativ linear sind (was bei technischen Kostendaten oft so ist), funktioniert das hervorragend.

🧾 Fazit:
→ Modell erklärt 80 % der Varianz.
→ Keine Regularisierung nötig — sehr gute baseline.
→ Evtl. etwas empfindlich gegenüber Ausreißern.

⚙️ Ridge Regression (L2-Regularisierung)

MAE = 2.90 → kleinster absoluter Fehler

R² = 0.77 → etwas weniger erklärte Varianz

Bestraft große Koeffizienten, dämpft Overfitting.

🧾 Fazit:
→ Stabiler, robuster, weniger anfällig für Multikollinearität.
→ Leicht schlechtere Gesamtanpassung (R²) — typisch für Ridge, da es „glättet“.
→ Wenn du Stabilität bevorzugst (z. B. in Produktion) → sehr gutes Modell.

⚙️ Lasso Regression (L1-Regularisierung)

MAE = 3.03, R² = 0.76

Macht viele Koeffizienten = 0 → führt Feature-Selektion durch.
→ Das erklärt, warum es etwas schwächer abschneidet: es hat einige erklärende Variablen komplett entfernt.

🧾 Fazit:
→ Gute Wahl, wenn du Interpretierbarkeit willst („welche Features sind wichtig“).
→ Nicht das beste in Performance, aber sehr hilfreich für Feature-Selektion.

⚙️ Elastic Net (L1 + L2 kombiniert)

Werte zwischen Ridge und Lasso

MAE = 2.97, R² = 0.77

Vereint Glättung und Feature-Selektion → robust, aber nicht Top in R².

🧾 Fazit:
→ Kompromiss zwischen Ridge & Lasso, aber bei deinen Daten kein Vorteil.
→ Solide, aber Linear und Ridge sind stärker.

🧮 3️⃣ Gesamtbewertung
Platz	Modell	R²	MAE	Kommentar
🥇	Linear Regression	0.800	3.60	Beste Gesamtanpassung – deine Daten sind fast linear
🥈	Ridge Regression	0.773	2.90	Stabiler, weniger Overfitting – leicht schwächer im R²
🥉	Elastic Net	0.770	2.97	Kompromiss, aber kein klarer Vorteil
🏅	Lasso	0.765	3.03	Feature-Selektion, dafür etwas weniger Leistung
🧠 4️⃣ Warum Linear Regression hier gewinnt

Deine Eingabedaten (z. B. Bauteilgewicht, Materialstärke, Fläche, One-Hot-encoded Bauteilnamen)
→ haben nahezu lineare Beziehungen zum Preis.

Regularisierung (Ridge/Lasso) ist nicht nötig, weil kein starkes Overfitting erkennbar ist.

Das lineare Modell kann diese Struktur direkt und unverzerrt abbilden.

Kurz gesagt:
👉 Deine Features sind sauber konstruiert und der Preis hängt fast linear von ihnen ab.
Darum bringt Regularisierung keinen Gewinn – sie glättet sogar nützliche Unterschiede weg.

💬 5️⃣ Empfehlung für den Projektkontext
Ziel	Empfehlung
Höchste Erklärkraft (Masterarbeit, SHAP, Interpretation)	✅ Linear Regression
Stabilität & Robustheit für Produktion	✅ Ridge Regression
Variable-Selektion / Feature-Reduktion	✅ Lasso Regression
Balance zwischen beiden Welten	✅ Elastic Net, aber kein Mehrwert hier
🧭 Fazit

👉 Dein bestes Modell ist aktuell die einfache Lineare Regression.
Sie erklärt 80 % der Preisvarianz mit einem durchschnittlichen Fehler von ~3.6 —
das ist für technische Kostendaten extrem stark und interpretierbar.

Die Regularisierungen sind in deinem Fall leicht zu stark (glätten zu sehr),
weil dein Datensatz schon sauber, klar strukturiert und ohne stark korrelierte Ausreißer ist.










[I 2025-10-13 16:24:17,784] Trial 49 finished with value: 6.30273499104181 and parameters: {'depth': 7, 'learning_rate': 0.10872036252401641, 'iterations': 478, 'l2_leaf_reg': 4.547339773238926, 'bagging_temperature': 0.5989386953486417, 'random_strength': 0.006493041020672167}. Best is trial 41 with value: 5.947699775465967.
Beste Hyperparameter: {'depth': 7, 'learning_rate': 0.1412265522021575, 'iterations': 695, 'l2_leaf_reg': 6.795535615252061, 'bagging_temperature': 0.5202750354050795, 'random_strength': 0.0692179004870684}
Mean Absolute Error: 4.345159028783259
Mean Squared Error: 194.603456418415
R2 Score: 0.49622086432419055
                              Feature Id  Importances
0                   Bauteilgewicht in kg    59.465397
1                                Flaeche    20.134549
2                                 Dichte     5.761928
3                 Anzahl der Abkantungen     4.258055
4          Breite Laserzuschnitt in mm 2     3.770948
5   Gesamtlänge aller Aussparungen in mm     3.161032
6          Breite Laserzuschnitt in mm 1     1.735505
7                    Materialsärke in mm     1.315686
8                                Bereich     0.258500
9                               Material     0.087476
10                          Bezugsquelle     0.040364
11                           Bezeichnung     0.010561
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\I764783\Desktop\github\costpilot\src\models\catboost.py", line 82, in <module>
AttributeError: module 'datetime' has no attribute 'now'



ChatGPT said:

Exzellent beobachtet, Abi 👏 — und das ist nicht ungewöhnlich!
Dass lineare Modelle (z. B. LinearRegression, Ridge, Lasso, ElasticNet) bei dir besser abschneiden als Decision Trees oder CatBoost, sagt sehr viel über die Struktur deiner Daten — und es ist ein starkes Signal, das du richtig interpretierst.

Lass uns das technisch und konzeptionell aufdröseln, damit du daraus ableiten kannst, was das über dein Problem aussagt und wie du weitermachst. 👇

🧠 1️⃣ Warum lineare Modelle manchmal besser sind
🔹 a) Deine Daten sind (überwiegend) linear

Wenn deine Zielvariable (z. B. Kosten, Preis o. Ä.) proportional oder additiv mit Features wie Gewicht, Fläche, Materialstärke usw. steigt,
dann ist der wahre Zusammenhang nahezu linear:

Kosten
≈
𝑎
×
Gewicht
+
𝑏
×
Fl
a
¨
che
+
𝑐
×
Materialst
a
¨
rke
+
…
Kosten≈a×Gewicht+b×Fl
a
¨
che+c×Materialst
a
¨
rke+…

→ Dann hat ein lineares Modell einen natürlichen Vorteil:
Es kann diesen Zusammenhang direkt, effizient und stabil lernen.

Ein Decision Tree oder CatBoost würde dagegen versuchen, diese glatte Beziehung in viele kleine „Stufen“ (Splits) zu zerhacken → das führt zu Overfitting oder Informationsverlust.

🔹 b) Wenig Daten / Rauschen

Du hast erwähnt, dass du (Train=115, Test=29) Beobachtungen hast — also relativ kleinen Datensatz.

Das bedeutet:

Bäume haben nicht genug Daten, um stabile Grenzwerte zu lernen.

CatBoost oder RandomForest haben viele Parameter → sie überparametrisieren leicht.

Lineare Modelle dagegen brauchen kaum Daten pro Feature → sie bleiben robust.

💡 Faustregel:

Bei < 500 Beobachtungen und „glatten“ Zusammenhängen schlagen lineare Modelle oft alles.

🔹 c) Features haben monotone, stetige Beziehungen

Wenn deine Features sowas sind wie:

Gewicht

Fläche

Anzahl Abkantungen

Dichte

Dann sind das alles metrische Variablen, die kontinuierlich mit der Zielgröße korrelieren.
→ Perfektes Terrain für lineare Regressoren.

Bäume sind am stärksten bei nichtlinearen, interaktiven Mustern:
z. B. „Wenn Material = Stahl UND Gewicht > 5 kg, dann ...“.
Wenn solche Schwellen kaum vorkommen, bringen Bäume keinen Mehrwert.

📈 2️⃣ Wie du das interpretieren solltest
Modell	R²	Interpretation
Linear Regression	Hoch (~0.8–0.9)	Deine Zusammenhänge sind linear & stabil
Decision Tree	Schwankend (~0.4–0.9)	Überfitting, instabil bei Splits
CatBoost	Mittel (~0.5)	Robust, aber kein komplexes Muster zu lernen
Fazit:		→ Lineares Modell genügt und ist das richtige Werkzeug! ✅