#include "EnvironmentControlPanel.h"

#ifdef _WIN32

#include <windows.h>
#include <commctrl.h>

#include <cstdio>

#pragma comment(lib, "Comctl32.lib")

namespace EnvironmentControlPanel {
namespace {

constexpr int TrackbarId = 1001;
constexpr int ResetButtonId = 1002;
constexpr int RotationStepsPerDegree = 10;
constexpr float DefaultRotationDegrees = 114.0f;

HWND panelWindow = nullptr;
HWND rotationTrackbar = nullptr;
HWND rotationLabel = nullptr;
float pendingRotationDegrees = DefaultRotationDegrees;
bool rotationChanged = false;

void UpdateRotationLabel(float degrees) {
    char text[64] = {};
    std::snprintf(text, sizeof(text), "HDRI Rotation: %.1f deg", degrees);
    SetWindowTextA(rotationLabel, text);
}

void SetRotation(float degrees, bool notifyRenderer) {
    pendingRotationDegrees = degrees;
    SendMessageA(rotationTrackbar, TBM_SETPOS, TRUE,
                 static_cast<LPARAM>(degrees * RotationStepsPerDegree));
    UpdateRotationLabel(degrees);
    if (notifyRenderer) {
        rotationChanged = true;
    }
}

LRESULT CALLBACK PanelWindowProcedure(HWND window, UINT message,
                                      WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_HSCROLL:
        if (reinterpret_cast<HWND>(lParam) == rotationTrackbar) {
            const LRESULT value = SendMessageA(
                rotationTrackbar, TBM_GETPOS, 0, 0);
            pendingRotationDegrees =
                static_cast<float>(value) / RotationStepsPerDegree;
            UpdateRotationLabel(pendingRotationDegrees);
            rotationChanged = true;
            return 0;
        }
        break;
    case WM_COMMAND:
        if (LOWORD(wParam) == ResetButtonId &&
            HIWORD(wParam) == BN_CLICKED) {
            SetRotation(DefaultRotationDegrees, true);
            return 0;
        }
        break;
    case WM_CLOSE:
        ShowWindow(window, SW_HIDE);
        return 0;
    case WM_DESTROY:
        panelWindow = nullptr;
        rotationTrackbar = nullptr;
        rotationLabel = nullptr;
        return 0;
    default:
        break;
    }
    return DefWindowProcA(window, message, wParam, lParam);
}

} // namespace

void Create(float initialRotationDegrees) {
    INITCOMMONCONTROLSEX controls = {};
    controls.dwSize = sizeof(controls);
    controls.dwICC = ICC_BAR_CLASSES;
    InitCommonControlsEx(&controls);

    HINSTANCE instance = GetModuleHandleA(nullptr);
    WNDCLASSA windowClass = {};
    windowClass.lpfnWndProc = PanelWindowProcedure;
    windowClass.hInstance = instance;
    windowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
    windowClass.hbrBackground =
        reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
    windowClass.lpszClassName = "VulkanEnvironmentControlPanel";
    RegisterClassA(&windowClass);

    panelWindow = CreateWindowExA(
        WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
        windowClass.lpszClassName, "Environment Controls",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
        40, 70, 390, 155, nullptr, nullptr, instance, nullptr);
    if (panelWindow == nullptr) {
        return;
    }

    rotationLabel = CreateWindowExA(
        0, "STATIC", "", WS_CHILD | WS_VISIBLE,
        18, 15, 245, 24, panelWindow, nullptr, instance, nullptr);
    rotationTrackbar = CreateWindowExA(
        0, TRACKBAR_CLASSA, "",
        WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS | TBS_HORZ,
        14, 42, 350, 42, panelWindow,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(TrackbarId)), instance,
        nullptr);
    SendMessageA(rotationTrackbar, TBM_SETRANGE, TRUE,
                 MAKELPARAM(0, 360 * RotationStepsPerDegree));
    SendMessageA(rotationTrackbar, TBM_SETTICFREQ,
                 30 * RotationStepsPerDegree, 0);

    CreateWindowExA(
        0, "BUTTON", "Reset", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        283, 88, 80, 27, panelWindow,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(ResetButtonId)),
        instance, nullptr);

    SetRotation(initialRotationDegrees, false);
    ShowWindow(panelWindow, SW_SHOW);
    UpdateWindow(panelWindow);
}

bool ConsumeRotation(float& rotationDegrees) {
    if (!rotationChanged) {
        return false;
    }
    rotationChanged = false;
    rotationDegrees = pendingRotationDegrees;
    return true;
}

float GetCurrentRotationDegrees() {
    return pendingRotationDegrees;
}

void Destroy() {
    if (panelWindow != nullptr) {
        DestroyWindow(panelWindow);
    }
}

} // namespace EnvironmentControlPanel

#else

namespace EnvironmentControlPanel {
void Create(float) {}
bool ConsumeRotation(float&) { return false; }
float GetCurrentRotationDegrees() { return 0.0f; }
void Destroy() {}
} // namespace EnvironmentControlPanel

#endif
