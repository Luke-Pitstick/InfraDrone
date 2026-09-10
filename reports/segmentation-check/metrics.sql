SELECT model, threshold, COUNT(*) AS images,
       1.0 * SUM(tp) / SUM(tp + fp + fn) AS iou,
       1.0 * SUM(tp) / SUM(tp + fp) AS precision,
       1.0 * SUM(tp) / SUM(tp + fn) AS recall
FROM clean_pixels
GROUP BY model, threshold
ORDER BY model, threshold;