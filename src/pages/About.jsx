import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function AboutPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-base text-base-color py-16">
      <div className="max-w-3xl mx-auto px-4">
        <Button
          variant="ghost"
          className="mb-6 gap-2 text-base-color hover:text-base-color"
          onClick={() => navigate(-1)}
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          Back
        </Button>
        <Card className="border-0 shadow-lg">
          <CardHeader>
            <CardTitle className="text-3xl font-bold text-base-color">
              About the Team
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6 text-muted-foreground text-lg">
            <section>
              <h2 className="text-xl font-semibold text-base-color mb-2">CEO &amp; Founder</h2>
              <p className="leading-relaxed">
                Mike Wofsey, Ph.D.
                <br />
                Research Physicist
              </p>
            </section>
            <section>
              <h2 className="text-xl font-semibold text-base-color mb-2">
                Project Leads and Developers
              </h2>
              <ul className="list-disc list-inside space-y-2">
                <li>Obi Alwani</li>
                <li>Habib Habib</li>
                <li>Nameera Nadeem</li>
                <li>David Lin</li>
                <li>Bruno Viera</li>
              </ul>
            </section>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
