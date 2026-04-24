import * as React from "react";
import { cn } from "../../lib/utils";

export const Card = React.forwardRef<HTMLElement, React.HTMLAttributes<HTMLElement>>(
  ({ className, ...props }, ref) => (
    <section
      ref={ref}
      data-slot="card"
      className={cn("overflow-hidden rounded-md border border-border bg-card text-card-foreground shadow-sm", className)}
      {...props}
    />
  ),
);
Card.displayName = "Card";

export const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} data-slot="card-header" className={cn("flex h-[42px] items-center justify-between gap-2.5 px-4", className)} {...props} />
  ),
);
CardHeader.displayName = "CardHeader";

export const CardTitle = React.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h2
      ref={ref}
      data-slot="card-title"
      className={cn("m-0 text-[13px] font-semibold uppercase tracking-[0.05em] text-foreground", className)}
      {...props}
    />
  ),
);
CardTitle.displayName = "CardTitle";

export const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} data-slot="card-content" className={cn("px-4 pb-4", className)} {...props} />
  ),
);
CardContent.displayName = "CardContent";
