const express = require("express");
const cors = require("cors");
const app = express();
app.use(cors());
app.use(express.json());

// ===== Wage Type Endpoints =====

app.post("/wagetype", (req, res) => {
    console.log("[MOCK] POST /wagetype:", JSON.stringify(req.body));
    const { wagetype, startdate, enddate } = req.body;

    const missing = [];
    if (!wagetype) missing.push("wagetype");
    if (!startdate) missing.push("startdate");
    if (!enddate) missing.push("enddate");

    if (missing.length > 0) {
        return res.status(400).json({
            status: "error",
            message: `Missing mandatory fields: ${missing.join(", ")}`,
        });
    }

    res.json({
        status: "success",
        message: `Wage type '${wagetype}' created successfully`,
        validFrom: startdate,
        validTo: enddate,
        data: req.body,
        timestamp: new Date().toISOString(),
    });
});

app.get("/getWageType", (req, res) => {
    console.log("[MOCK] GET /getWageType:", JSON.stringify(req.query));
    const wt = req.query.wagetype;
    
    const db = {
        "0001": { description: "Regular Salary", status: "active", country: "US", group: "Base Pay" },
        "0002": { description: "Annual Bonus", status: "active", country: "US", group: "Additional Pay" },
        "1001": { description: "Housing Allowance", status: "active", country: "US", group: "Allowances" },
        "5000": { description: "Health Insurance Deduction", status: "restricted", country: "DE", group: "Deductions" }
    };

    const data = db[wt] || { description: "Custom Wage Type", status: "unknown", country: "Global", group: "General" };

    res.json({
        wagetype: wt || "0000",
        ...data,
        createdAt: "2024-01-15T10:30:00Z",
    });
});

app.get("/wagetyperelated", (req, res) => {
    console.log("[MOCK] GET /wagetyperelated:", JSON.stringify(req.query));
    const wt = req.query.wagetype;

    const relations = {
        "0001": [
            { relationType: "pension_eligible", relatedWagetype: "9000", description: "Pension Base" },
            { relationType: "tax_reference", relatedWagetype: "T001", description: "Federal Tax" }
        ],
        "1001": [
            { relationType: "linked_base", relatedWagetype: "0001", description: "Calculated on base" }
        ]
    };

    res.json({
        wagetype: wt || "0000",
        relatedData: relations[wt] || [
            { relationType: "generic", relatedWagetype: "9999", description: "Standard mapping" }
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
const notifications = [];

app.get("/api/notifications", (req, res) => {
    res.json(notifications);
});

app.get("/notification", (req, res) => {
    console.log("[MOCK] GET /notification:", JSON.stringify(req.query));
    const notif = {
        id: req.query.notificationId || `N-${Date.now()}`,
        details: req.query.notifDetails || "",
        sentAt: new Date().toISOString(),
        method: "GET"
    };
    notifications.unshift(notif);
    res.json({ status: "sent", ...notif });
});

app.post("/notification", (req, res) => {
    console.log("[MOCK] POST /notification:", JSON.stringify(req.body));
    const notif = {
        id: req.body.notificationId || `N-${Date.now()}`,
        details: req.body.notifDetails || "",
        sentAt: new Date().toISOString(),
        method: "POST"
    };
    notifications.unshift(notif);
    res.json({ status: "sent", ...notif });
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