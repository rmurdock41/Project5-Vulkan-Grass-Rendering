#pragma once

namespace EnvironmentControlPanel {

// Opens a small native control window. The renderer remains independent from
// the UI; main.cpp consumes changes and forwards them to the environment UBO.
void Create(float initialRotationDegrees);
bool ConsumeRotation(float& rotationDegrees);
float GetCurrentRotationDegrees();
void Destroy();

} // namespace EnvironmentControlPanel
