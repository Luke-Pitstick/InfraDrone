import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded px-2 py-1 text-[11px] font-bold leading-none transition-colors",
  {
  variants: {
    variant: {
      default: "border border-border bg-secondary text-secondary-foreground",
      success: "border border-green-200 bg-green-50 text-green-700",
      danger: "border border-red-200 bg-red-50 text-red-700",
      warning: "border border-amber-200 bg-amber-50 text-amber-700",
      muted: "border border-border bg-muted text-muted-foreground",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

export type BadgeProps = React.HTMLAttributes<HTMLSpanElement> &
  VariantProps<typeof badgeVariants>;

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant, ...props }, ref) => (
    <span ref={ref} data-slot="badge" className={cn(badgeVariants({ variant }), className)} {...props} />
  ),
);
Badge.displayName = "Badge";
