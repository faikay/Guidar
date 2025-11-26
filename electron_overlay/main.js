const { app, BrowserWindow, screen } = require('electron');
const path = require('path');

function createWindow() {
    const { width, height } = screen.getPrimaryDisplay().workAreaSize;

    const mainWindow = new BrowserWindow({
        width: 400,
        height: 400,
        x: width - 450, // Position bottom right by default, or change as needed
        y: height - 450,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        },
        transparent: true,
        frame: false,
        alwaysOnTop: true,
        hasShadow: false,
        resizable: false,
        skipTaskbar: true,
        type: 'toolbar'
    });

    mainWindow.setAlwaysOnTop(true, 'screen-saver');
    mainWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

    mainWindow.loadFile('index.html');
    
    // Optional: Open the DevTools.
    // mainWindow.webContents.openDevTools({ mode: 'detach' });

    // Make the window click-through (ignore mouse events)
    // If you want to be able to drag it, you need to handle that in the renderer
    // and toggle this. For a pure overlay, we often want click-through.
    // mainWindow.setIgnoreMouseEvents(true, { forward: true });
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});
