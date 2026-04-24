import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md border text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
  variants: {
    variant: {
      default: "border-primary bg-primary text-primary-foreground hover:bg-primary/90",
      destructive: "border-destructive bg-destructive/10 text-destructive hover:bg-destructive hover:text-destructive-foreground",
      outline: "border-input bg-background text-primary hover:bg-accent hover:text-accent-foreground",
      ghost: "border-border bg-background/90 text-foreground shadow-sm hover:bg-accent hover:text-accent-foreground",
    },
    size: {
      default: "h-9 min-w-28 px-4 py-2",
      icon: "h-9 w-9 p-0",
    },
  },
  defaultVariants: {
    variant: "default",
    size: "default",
  },
});

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants>;

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button ref={ref} data-slot="button" className={cn(buttonVariants({ variant, size }), className)} {...props} />
  ),
);
Button.displayName = "Button";
