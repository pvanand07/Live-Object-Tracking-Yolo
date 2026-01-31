/**
 * WebSocket Video Stream Client
 * Connects to FastAPI server and displays real-time object tracking
 */

class VideoStreamClient {
    constructor() {
        // WebSocket connection
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.isManualDisconnect = false;

        // Canvas and context
        this.canvas = document.getElementById('videoCanvas');
        this.ctx = this.canvas.getContext('2d');

        // DOM elements
        this.elements = {
            connectionStatus: document.getElementById('connectionStatus'),
            videoOverlay: document.getElementById('videoOverlay'),
            fps: document.getElementById('fps'),
            frameCount: document.getElementById('frameCount'),
            latency: document.getElementById('latency'),
            objectCount: document.getElementById('objectCount'),
            configInfo: document.getElementById('configInfo'),
            connectBtn: document.getElementById('connectBtn'),
            disconnectBtn: document.getElementById('disconnectBtn')
        };

        // Stats
        this.frameReceived = 0;
        this.lastFrameTime = Date.now();

        // Event listeners
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.elements.connectBtn.addEventListener('click', () => this.connect());
        this.elements.disconnectBtn.addEventListener('click', () => this.disconnect());
    }

    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('Already connected');
            return;
        }

        this.isManualDisconnect = false;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        console.log(`Connecting to ${wsUrl}...`);
        this.updateConnectionStatus('connecting', 'Connecting...');

        try {
            this.ws = new WebSocket(wsUrl);
            this.ws.onopen = this.onOpen.bind(this);
            this.ws.onmessage = this.onMessage.bind(this);
            this.ws.onerror = this.onError.bind(this);
            this.ws.onclose = this.onClose.bind(this);
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.updateConnectionStatus('error', 'Connection Failed');
        }
    }

    disconnect() {
        this.isManualDisconnect = true;
        if (this.ws) {
            this.ws.close();
        }
        this.updateConnectionStatus('disconnected', 'Disconnected');
        this.elements.connectBtn.disabled = false;
        this.elements.disconnectBtn.disabled = true;
    }

    onOpen(event) {
        console.log('✓ WebSocket connected');
        this.reconnectAttempts = 0;
        this.updateConnectionStatus('connected', 'Connected');
        this.elements.connectBtn.disabled = true;
        this.elements.disconnectBtn.disabled = false;
        this.elements.videoOverlay.classList.add('hidden');
        this.canvas.classList.add('active');
    }

    onMessage(event) {
        try {
            const data = JSON.parse(event.data);

            switch (data.type) {
                case 'config':
                    this.displayConfig(data.data);
                    break;

                case 'frame':
                    this.displayFrame(data.image, data.metadata);
                    break;

                case 'error':
                    console.error('Server error:', data.message);
                    break;

                default:
                    console.warn('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error processing message:', error);
        }
    }

    onError(event) {
        console.error('WebSocket error:', event);
        this.updateConnectionStatus('error', 'Connection Error');
    }

    onClose(event) {
        console.log('WebSocket closed:', event.code, event.reason);
        this.canvas.classList.remove('active');
        this.elements.videoOverlay.classList.remove('hidden');
        this.elements.connectBtn.disabled = false;
        this.elements.disconnectBtn.disabled = true;

        // Auto-reconnect if not manual disconnect
        if (!this.isManualDisconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            this.updateConnectionStatus('reconnecting', `Reconnecting (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.connect(), delay);
        } else {
            this.updateConnectionStatus('disconnected', 'Disconnected');
        }
    }

    displayFrame(base64Image, metadata) {
        const img = new Image();

        img.onload = () => {
            // Resize canvas to match image
            if (this.canvas.width !== img.width || this.canvas.height !== img.height) {
                this.canvas.width = img.width;
                this.canvas.height = img.height;
            }

            // Draw image on canvas
            this.ctx.drawImage(img, 0, 0);

            // Update stats
            this.updateStats(metadata);

            // Calculate latency
            const now = Date.now();
            const latency = now - this.lastFrameTime;
            this.lastFrameTime = now;
            this.elements.latency.textContent = `${latency} ms`;

            this.frameReceived++;
        };

        img.onerror = (error) => {
            console.error('Error loading image:', error);
        };

        img.src = `data:image/jpeg;base64,${base64Image}`;
    }

    updateStats(metadata) {
        this.elements.fps.textContent = metadata.fps || 0;
        this.elements.frameCount.textContent = metadata.frame_count || 0;
        this.elements.objectCount.textContent = metadata.num_detections || 0;
    }



    displayConfig(config) {
        const html = `
            <p><strong>Host:</strong> ${config.host}</p>
            <p><strong>Port:</strong> ${config.port}</p>
            <p><strong>Resolution:</strong> ${config.camera_resolution}</p>
            <p><strong>Target FPS:</strong> ${config.target_fps}</p>
            <p><strong>YOLO Model:</strong> ${config.yolo_model}</p>
            <p><strong>JPEG Quality:</strong> ${config.jpeg_quality}%</p>
        `;
        this.elements.configInfo.innerHTML = html;
    }

    updateConnectionStatus(status, text) {
        const statusDot = this.elements.connectionStatus.querySelector('.status-dot');
        const statusText = this.elements.connectionStatus.querySelector('.status-text');

        statusText.textContent = text;

        // Update dot color
        statusDot.classList.remove('connected');
        if (status === 'connected') {
            statusDot.classList.add('connected');
        }
    }
}

// Initialize client when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing Video Stream Client...');
    const client = new VideoStreamClient();

    // Auto-connect on load
    setTimeout(() => client.connect(), 500);
});
