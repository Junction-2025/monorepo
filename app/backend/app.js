import express from "express";
import cors from "cors";
const app = express();
const PORT = 8000;

app.use(cors());

// Middleware to parse JSON
app.use(express.json());

// Example route
app.get("/", (req, res) => {
    res.send("Hello from Express!");
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});