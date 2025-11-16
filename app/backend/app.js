import express from "express";
import cors from "cors";
import EventEmitter from "events"
import { runAgent } from "./agent/agent.js";

const app = express();
const PORT = 8000;

app.use(cors());

// Middleware to parse JSON
app.use(express.json());

const eventEmitter = new EventEmitter();

export async function sendSSE(data) {
    console.log("sending dta via sse");
    console.log(data);
    eventEmitter.emit("sendEvent", data);
};

app.post('/agent', async (req, res) => {
    const data = req.body;
    const result = await runAgent(data);
    res.json({ success: true, message: result });
});

app.post('/trigger_event', async (req, res) => {
    const droneData = req.body;
    /*
    {
        type: "drone-detected",
        model: "DJI M600",
        rpm: "5000",
        rotors: 4,
        description: "Target detected via event camera system. Classification: Commercial hexacopter. Status: Active monitoring. Last detected: 0.5 seconds ago. Confidence: 94.7%",
        timestamp: new Date().toISOString()
    }
    */

    // Send to all connected SSE clients
    await sendSSE(droneData);
    //await sendSSE(droneData);

    res.json({ success: true, message: "Drone data sent to clients" });
})

app.get('/events', async (req, res) => {
    // Set headers to keep the connection alive and tell the client we're sending event-stream data
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    // Send an initial message
    console.log("event connected");
    res.write(`data: ${JSON.stringify({ message: "Connected to server" })}\n\n`);

    const userSendEvents = (data) => {
        res.write(`data: ${JSON.stringify(data)}\n\n`);
    };

    eventEmitter.on('sendEvent', userSendEvents);

    await sendSSE({
        type: "drone-detected",
        model: "DJI M600",
        rpm: "5000",
        rotors: 6,
        description: "Target detected via event camera system. Classification: Commercial hexacopter. Status: Active monitoring. Last detected: 0.5 seconds ago. Confidence: 94.7%",
        timestamp: new Date().toLocaleString()
    });

    req.on('close', () => {
        eventEmitter.off('sendEvent', userSendEvents);
        res.end();
    });
});

// Example route
app.get("/", (req, res) => {
    res.send("Hello from Express!");
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});

