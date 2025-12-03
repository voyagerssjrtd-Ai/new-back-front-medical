from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from run import calling_langgarph
from typing import List,Optional, Dict, Any
from pydantic import BaseModel
import sqlite3
import os
from pathlib import Path
import utils.security as security

DB_FILE = "triage.db"
app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Change to ["http://localhost:3000"] for React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running"}


@app.post("/chat")
def chat_endpoint(user_input: str):
    response = calling_langgarph(user_input.query)
    # Return as JSON object (not string)
    return json.loads(response)


# File Uploads
@app.post("/saveFiles")
async def uploadDocuments(file: UploadFile = File(...)):
    try:
        os.makedirs("savedFiles", exist_ok=True)
        with open(f'savedFiles/{file.filename}', "wb") as f:
            f.write(await file.read())
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/saveMultipleFiles")
async def uploadDocuments(files: List[UploadFile] = File(...)):
    try:
        os.makedirs("savedFiles", exist_ok=True)
        saved_files = []
        for file in files:
            with open(f'savedFiles/{file.filename}', "wb") as f:
                f.write(await file.read())
            saved_files.append(file.filename)
        return {"status": "success", "files": saved_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Doctor functions
# ---------------------------
def query_db(query: str, params: tuple = ()) -> list[Dict[str, Any]]:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def execute_db(query: str, params: tuple = ()):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()

# ---------------------------
# Pydantic Models
# ---------------------------
class Doctor(BaseModel):
    name: str
    specialty: str
    department: str
    location: str

class DoctorUpdate(BaseModel):
    name: Optional[str] = None
    specialty: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None

# ---------------------------
# Endpoints for Doctors
# ---------------------------
@app.get("/doctors")
def get_all_doctors():
    return query_db("SELECT * FROM doctors")

@app.get("/doctors/{doctor_id}")
def get_doctor_by_id(doctor_id: int):
    records = query_db("SELECT * FROM doctors WHERE id = ?", (doctor_id,))
    if not records:
        raise HTTPException(status_code=404, detail="Doctor not found")
    return records[0]

@app.post("/doctors")
def create_doctor(doctor: Doctor):
    execute_db(
        "INSERT INTO doctors (name, specialty, department, location) VALUES (?, ?, ?, ?)",
        (doctor.name, doctor.specialty, doctor.department, doctor.location),
    )
    return {"message": "Doctor created successfully"}

@app.delete("/doctors/{doctor_id}")
def delete_doctor(doctor_id: int):
    execute_db("DELETE FROM doctors WHERE id = ?", (doctor_id,))
    return {"message": "Doctor deleted successfully"}

@app.patch("/doctors/{doctor_id}")
def partial_update_doctor(doctor_id: int, doctor: DoctorUpdate):
    # Build dynamic update query
    fields = []
    values = []
    for field, value in doctor.dict(exclude_unset=True).items():
        fields.append(f"{field} = ?")
        values.append(value)

    if not fields:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    values.append(doctor_id)
    query = f"UPDATE doctors SET {', '.join(fields)} WHERE id = ?"
    execute_db(query, tuple(values))

    return {"message": "Doctor updated successfully"}

# ---------------------------
# Labs
# ---------------------------
class Lab(BaseModel):
    name: str
    location: str

class LabUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None

@app.get("/labs")
def get_all_labs():
    return query_db("SELECT * FROM labs")

@app.get("/labs/{lab_id}")
def get_lab_by_id(lab_id: int):
    records = query_db("SELECT * FROM labs WHERE id = ?", (lab_id,))
    if not records:
        raise HTTPException(status_code=404, detail="Lab not found")
    return records[0]

@app.post("/labs")
def create_lab(lab: Lab):
    execute_db("INSERT INTO labs (name, location) VALUES (?, ?)", (lab.name, lab.location))
    return {"message": "Lab created successfully"}

@app.delete("/labs/{lab_id}")
def delete_lab(lab_id: int):
    execute_db("DELETE FROM labs WHERE id = ?", (lab_id,))
    return {"message": "Lab deleted successfully"}

@app.patch("/labs/{lab_id}")
def partial_update_lab(lab_id: int, lab: LabUpdate):
    fields, values = [], []
    for field, value in lab.dict(exclude_unset=True).items():
        fields.append(f"{field} = ?")
        values.append(value)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields provided for update")
    values.append(lab_id)
    query = f"UPDATE labs SET {', '.join(fields)} WHERE id = ?"
    execute_db(query, tuple(values))
    return {"message": "Lab updated successfully"}

# ---------------------------
# Availability
# ---------------------------
class Availability(BaseModel):
    resource_type: str
    resource_id: int
    slot_start: str
    slot_end: str
    is_available: int

class AvailabilityUpdate(BaseModel):
    resource_type: Optional[str] = None
    resource_id: Optional[int] = None
    slot_start: Optional[str] = None
    slot_end: Optional[str] = None
    is_available: Optional[int] = None

@app.get("/availability")
def get_all_availability():
    return query_db("SELECT * FROM availability")

@app.get("/availability/{avail_id}")
def get_availability_by_id(avail_id: int):
    records = query_db("SELECT * FROM availability WHERE id = ?", (avail_id,))
    if not records:
        raise HTTPException(status_code=404, detail="Availability not found")
    return records[0]

@app.post("/availability")
def create_availability(avail: Availability):
    execute_db(
        "INSERT INTO availability (resource_type, resource_id, slot_start, slot_end, is_available) VALUES (?, ?, ?, ?, ?)",
        (avail.resource_type, avail.resource_id, avail.slot_start, avail.slot_end, avail.is_available),
    )
    return {"message": "Availability created successfully"}

@app.delete("/availability/{avail_id}")
def delete_availability(avail_id: int):
    execute_db("DELETE FROM availability WHERE id = ?", (avail_id,))
    return {"message": "Availability deleted successfully"}

@app.patch("/availability/{avail_id}")
def partial_update_availability(avail_id: int, avail: AvailabilityUpdate):
    fields, values = [], []
    for field, value in avail.dict(exclude_unset=True).items():
        fields.append(f"{field} = ?")
        values.append(value)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields provided for update")
    values.append(avail_id)
    query = f"UPDATE availability SET {', '.join(fields)} WHERE id = ?"
    execute_db(query, tuple(values))
    return {"message": "Availability updated successfully"}

# ---------------------------
# Appointments (Bookings)
# ---------------------------
class Appointment(BaseModel):
    user_id: str
    kind: str
    resource_id: Optional[int] = None
    resource_type: Optional[str] = None
    requested_slot: Optional[str] = None
    booked_slot: Optional[str] = None
    status: str
    suggested_alternatives: Optional[str] = None

class AppointmentUpdate(BaseModel):
    user_id: Optional[str] = None
    kind: Optional[str] = None
    resource_id: Optional[int] = None
    resource_type: Optional[str] = None
    requested_slot: Optional[str] = None
    booked_slot: Optional[str] = None
    status: Optional[str] = None
    suggested_alternatives: Optional[str] = None

@app.get("/appointments")
def get_all_appointments():
    return query_db("SELECT * FROM appointments")

@app.get("/appointments/{appointment_id}")
def get_appointment_by_id(appointment_id: int):
    records = query_db("SELECT * FROM appointments WHERE id = ?", (appointment_id,))
    if not records:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return records[0]

@app.post("/appointments")
def create_appointment(appt: Appointment):
    execute_db(
        """INSERT INTO appointments 
        (user_id, kind, resource_id, resource_type, requested_slot, booked_slot, status, suggested_alternatives) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (appt.user_id, appt.kind, appt.resource_id, appt.resource_type, appt.requested_slot,
         appt.booked_slot, appt.status, appt.suggested_alternatives),
    )
    return {"message": "Appointment created successfully"}

@app.delete("/appointments/{appointment_id}")
def delete_appointment(appointment_id: int):
    execute_db("DELETE FROM appointments WHERE id = ?", (appointment_id,))
    return {"message": "Appointment deleted successfully"}

@app.patch("/appointments/{appointment_id}")
def partial_update_appointment(appointment_id: int, appt: AppointmentUpdate):
    fields, values = [], []
    for field, value in appt.dict(exclude_unset=True).items():
        fields.append(f"{field} = ?")
        values.append(value)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields provided for update")
    values.append(appointment_id)
    query = f"UPDATE appointments SET {', '.join(fields)} WHERE id = ?"
    execute_db(query, tuple(values))
    return {"message": "Appointment updated successfully"}




