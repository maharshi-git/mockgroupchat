const express = require("express");
const app = express();
app.use(express.json());

// ===== Wage Type Endpoints =====

app.post("/wagetype", (req, res) => {
    console.log("[MOCK] POST /wagetype:", JSON.stringify(req.body));
    res.json({
        status: "success",
        message: `Wage type '${req.body.wagetype || "unknown"}' created successfully`,
        data: req.body,
        timestamp: new Date().toISOString(),
    });
});

app.get("/getWageType", (req, res) => {
    console.log("[MOCK] GET /getWageType:", JSON.stringify(req.query));
    res.json({
        wagetype: req.query.wagetype || "0000",
        description: "Monthly Base Salary",
        status: "active",
        country: "US",
        createdAt: "2024-01-15T10:30:00Z",
    });
});

app.get("/wagetyperelated", (req, res) => {
    console.log("[MOCK] GET /wagetyperelated:", JSON.stringify(req.query));
    res.json({
        wagetype: req.query.wagetype || "0000",
        relatedData: [
            { relationType: "base_wage", relatedWagetype: "1001", description: "Base component" },
            { relationType: "tax_wage", relatedWagetype: "2001", description: "Tax component" },
        ],
    });
});

// ===== ServiceNow Endpoints =====

app.get("/getsnowdet", (req, res) => {
    console.log("[MOCK] GET /getsnowdet:", JSON.stringify(req.query));
    res.json({
        SNOWNumber: req.query.SNOWNumber || "INC000",
        status: "open",
        priority: "2 - High",
        description: "Wage type configuration issue",
        assignedTo: "Admin",
        createdAt: "2024-03-01T08:00:00Z",
    });
});

app.post("/closesnow", (req, res) => {
    console.log("[MOCK] POST /closesnow:", JSON.stringify(req.body));
    res.json({
        status: "success",
        message: `SNOW incident ${req.body.SNOWNumber} closed`,
        closeRemarks: req.body.CloseRemarks,
        closedAt: new Date().toISOString(),
    });
});

app.post("/updatesnow", (req, res) => {
    console.log("[MOCK] POST /updatesnow:", JSON.stringify(req.body));
    res.json({
        status: "success",
        message: `SNOW incident ${req.body.SNOWNumber} updated`,
        updateRemarks: req.body.UpdateRemarks,
        updatedAt: new Date().toISOString(),
    });
});

// ===== Notification Endpoints =====

app.get("/notification", (req, res) => {
    console.log("[MOCK] GET /notification:", JSON.stringify(req.query));
    res.json({
        status: "sent",
        notificationId: req.query.notificationId || "N/A",
        notifDetails: req.query.notifDetails || "",
        sentAt: new Date().toISOString(),
    });
});

app.post("/notification", (req, res) => {
    console.log("[MOCK] POST /notification:", JSON.stringify(req.body));
    res.json({
        status: "sent",
        notificationId: req.body.notificationId || "N/A",
        notifDetails: req.body.notifDetails || "",
        sentAt: new Date().toISOString(),
    });
});

// ===== Generic REST (for ExternalRestService / P2PAgent) =====

app.get("/posts", (req, res) => {
    console.log("[MOCK] GET /posts:", JSON.stringify(req.query));
    res.json([
        {
            id: parseInt(req.query.id) || 1,
            title: "Test Post",
            body: "This is a test post body",
            userId: 1,
        },
    ]);
});

// ===== Start =====

const PORT = 3001;
app.listen(PORT, () => {
    console.log(`\n🔧 Mock Server running on http://localhost:${PORT}`);
    console.log(`   Endpoints:`);
    console.log(`   POST /wagetype         - Create wage type`);
    console.log(`   GET  /getWageType      - Read wage type`);
    console.log(`   GET  /wagetyperelated  - Read wage type related data`);
    console.log(`   GET  /getsnowdet       - Read SNOW incident`);
    console.log(`   POST /closesnow        - Close SNOW incident`);
    console.log(`   POST /updatesnow       - Update SNOW incident`);
    console.log(`   GET  /notification     - Send/read notification`);
    console.log(`   POST /notification     - Send notification`);
    console.log(`   GET  /posts            - Read posts (REST)\n`);
});