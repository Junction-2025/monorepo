'use client'
import { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from '@/components/ui/card';
import { AlertCircle, DroneIcon } from 'lucide-react';
import { Radar } from 'lucide-react';
import { Drone } from 'lucide-react';
import { ShieldAlert } from 'lucide-react';

export default function Dashboard() {


    const rpm = 2800;
    const rotors = 4;
    const model = "DJI M600"
    const description = "Target detected via event camera system. Classification: Commercial hexacopter. Status: Active monitoring. Last detected: 2.3 seconds ago. Confidence: 94.7%";

    return (
        <div className="flex flex-col items-center justify-center h-screen w-screen gap-12">
            <h1 className="text-4xl font-semibold">Drone Destroyer 5000</h1>
            <Card className="max-w-80">
                <CardHeader>
                    <CardTitle className="flex flex-row gap-2 items-center text-xl">
                        <Drone className="w-6 h-6 shrink-0" />
                        Detection
                    </CardTitle>
                </CardHeader>
                <CardContent className="text-sm border px-4 py-4 rounded-xl mx-4">
                    <ul className="flex flex-col list-disc list-inside gap-2">
                        <li>Model: {model}</li>
                        <li>Rotors: {rotors}</li>
                        <li>RPM: {rpm}</li>
                    </ul>
                </CardContent>
                <CardFooter className="flex text-sm text-muted-foreground items-start gap-2">
                    <ShieldAlert className="w-4 h-4 mt-1 shrink-0" />
                    <p className="flex-1">{description}</p>
                </CardFooter>
            </Card>
        </div>
    );
}