import e from "express";
import cookieParser from "cookie-parser";

const app = e();

app.use(e.json());
app.use(cookieParser());

export default app;